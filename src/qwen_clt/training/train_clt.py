from __future__ import annotations

import math
from pathlib import Path
import json
import torch
from tqdm import tqdm

from qwen_clt.data.text_dataset import iter_token_batches, iter_eval_batches
from qwen_clt.models.qwen_hooks import load_qwen_model_and_tokenizer, QwenMLPHookCollector
from qwen_clt.models.cross_layer_transcoder import (
    CrossLayerTranscoder,
    clt_normalization_kwargs,
)
from qwen_clt.training.losses import (
    reconstruction_loss,
    tanh_sparsity_loss,
    build_layer_sparsity_weights,
)
from qwen_clt.training.metrics import summarize_metrics
from qwen_clt.utils.seed import set_seed
from qwen_clt.utils.io import save_checkpoint


# ---------------------------------------------------------------------------
# LR scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup for `warmup_steps`, then cosine decay to `min_lr_ratio * base_lr`."""
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0. Got {warmup_steps}.")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0. Got {total_steps}.")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1]. Got {min_lr_ratio}.")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # linear warmup from 0 to 1
            return float(step + 1) / float(max(warmup_steps, 1))
        # cosine decay from 1 to min_lr_ratio
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = float(step - warmup_steps) / float(decay_steps)
        progress = min(progress, 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Sparsity lambda schedule: linear warmup from 0 to target lambda
# ---------------------------------------------------------------------------

def get_lambda_sparsity(
    step: int,
    *,
    target_lambda: float,
    warmup_steps: int,
) -> float:
    """Ramp sparsity penalty from 0 to target_lambda over warmup_steps."""
    if warmup_steps <= 0:
        return target_lambda
    return target_lambda * min(1.0, float(step) / float(warmup_steps))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def resolve_step_limits(training_cfg: dict, grad_accum: int) -> tuple[int, int | None]:
    if "max_optimizer_steps" in training_cfg:
        max_optimizer_steps = int(training_cfg["max_optimizer_steps"])
        max_micro_steps = training_cfg.get("max_micro_steps")
        if max_micro_steps is not None:
            max_micro_steps = int(max_micro_steps)
    elif "max_steps" in training_cfg:
        max_micro_steps = int(training_cfg["max_steps"])
        max_optimizer_steps = max_micro_steps // grad_accum
        print(
            "[training warning] training.max_steps is legacy and is interpreted "
            "as a micro-batch limit. Prefer training.max_optimizer_steps."
        )
    else:
        raise KeyError(
            "Missing training.max_optimizer_steps. "
            "Legacy training.max_steps is still accepted as a micro-batch limit."
        )

    if max_optimizer_steps <= 0:
        raise ValueError(
            f"max_optimizer_steps must be positive. Got {max_optimizer_steps}."
        )
    if max_micro_steps is not None and max_micro_steps <= 0:
        raise ValueError(f"max_micro_steps must be positive. Got {max_micro_steps}.")

    return max_optimizer_steps, max_micro_steps


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return -1


def prune_old_step_checkpoints(output_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return

    checkpoints = sorted(
        output_dir.glob("clt_step_*.pt"),
        key=_checkpoint_step,
    )
    stale = checkpoints[:-keep_last]

    for path in stale:
        path.unlink()


def load_initial_clt_weights(
    clt: CrossLayerTranscoder,
    checkpoint_path: str | Path,
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initial CLT checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected checkpoint dict at {checkpoint_path}, got {type(checkpoint)}."
        )
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint {checkpoint_path} has no 'model_state_dict'. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    clt.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


# ---------------------------------------------------------------------------
# Eval on held-out split
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_clt(
    clt: CrossLayerTranscoder,
    collector: QwenMLPHookCollector,
    cfg: dict,
    tokenizer,
    device: str,
) -> dict:
    """Run reconstruction eval on the validation split.

    Returns a dict with eval_nmse_mean, eval_l0_mean, eval_nmse_by_layer,
    eval_l0_by_layer.
    """
    clt.eval()
    all_features: list[list[torch.Tensor]] = []
    all_recons: list[list[torch.Tensor]] = []
    all_targets: list[list[torch.Tensor]] = []

    for batch in iter_eval_batches(cfg, tokenizer, device=device):
        acts = collector.run(batch.input_ids, batch.attention_mask)
        features, recons = clt(acts.mlp_inputs)
        all_features.append([f.detach().cpu() for f in features])
        all_recons.append([r.detach().cpu() for r in recons])
        all_targets.append([t.detach().cpu() for t in acts.mlp_outputs])

    clt.train()

    if not all_features:
        return {}

    n_layers = len(all_features[0])
    # average per-layer metrics across batches
    nmse_by_layer: list[float] = []
    l0_by_layer: list[float] = []

    from qwen_clt.training.losses import normalized_mse

    for layer in range(n_layers):
        layer_recons = torch.cat([b[layer] for b in all_recons], dim=1)
        layer_targets = torch.cat([b[layer] for b in all_targets], dim=1)
        layer_features = torch.cat([b[layer] for b in all_features], dim=1)

        nmse_by_layer.append(float(normalized_mse(layer_recons, layer_targets).item()))
        l0_by_layer.append(float((layer_features > 0).float().sum(dim=-1).mean().item()))

    return {
        "eval_nmse_mean": sum(nmse_by_layer) / len(nmse_by_layer),
        "eval_l0_mean": sum(l0_by_layer) / len(l0_by_layer),
        "eval_nmse_by_layer": nmse_by_layer,
        "eval_l0_by_layer": l0_by_layer,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_clt(cfg: dict) -> Path:
    seed = int(cfg["training"].get("seed", 42))
    set_seed(seed)
    device = cfg["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_qwen_model_and_tokenizer(cfg)
    collector = QwenMLPHookCollector(model)

    clt_cfg = cfg["clt"]
    clt = CrossLayerTranscoder(
        n_layers=int(clt_cfg["n_layers"]),
        d_model=int(clt_cfg["d_model"]),
        features_per_layer=int(clt_cfg["features_per_layer"]),
        init_threshold=float(clt_cfg.get("init_threshold", 0.0)),
        decoder_init_scale=float(clt_cfg.get("decoder_init_scale", 0.02)),
        **clt_normalization_kwargs(clt_cfg),
    ).to(device)

    training_cfg = cfg["training"]
    init_checkpoint_path = training_cfg.get("init_from_checkpoint")
    init_checkpoint = None
    if init_checkpoint_path:
        init_checkpoint = load_initial_clt_weights(clt, init_checkpoint_path)
        clt.to(device)

    optimizer = torch.optim.AdamW(
        clt.parameters(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )

    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 1))
    if grad_accum <= 0:
        raise ValueError(
            f"gradient_accumulation_steps must be positive. Got {grad_accum}."
        )

    max_optimizer_steps, max_micro_steps = resolve_step_limits(
        training_cfg,
        grad_accum=grad_accum,
    )

    # LR scheduler
    scheduler_cfg = training_cfg.get("lr_scheduler") or {}
    warmup_steps = int(scheduler_cfg.get("warmup_steps", 0))
    min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.1))
    use_scheduler = bool(scheduler_cfg.get("enabled", warmup_steps > 0))

    scheduler = build_lr_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=max_optimizer_steps,
        min_lr_ratio=min_lr_ratio,
    ) if use_scheduler else None

    # Restore scheduler state if continuing from checkpoint
    if scheduler is not None and init_checkpoint is not None:
        saved_scheduler = init_checkpoint.get("scheduler_state_dict")
        if saved_scheduler is not None:
            scheduler.load_state_dict(saved_scheduler)
            print(f"[train_clt] Restored scheduler state from checkpoint.")

    log_every = int(training_cfg.get("log_every", 50))
    eval_every = int(training_cfg.get("eval_every", 0))
    save_every = int(training_cfg.get("save_every", 1000))
    keep_last_checkpoints = int(training_cfg.get("keep_last_checkpoints", 0))
    lambda_sparsity_target = float(training_cfg.get("lambda_sparsity", 1e-4))
    sparsity_warmup_steps = int(training_cfg.get("sparsity_warmup_steps", 0))
    sparsity_c = float(training_cfg.get("sparsity_c", 1.0))
    grad_clip = float(training_cfg.get("grad_clip_norm", 1.0))

    # Per-layer sparsity weights: deep layers get higher pressure to stay sparse.
    # Controlled by training.layer_sparsity_weights (mode/min/max).
    lsw_cfg = training_cfg.get("layer_sparsity_weights") or {}
    layer_weights = build_layer_sparsity_weights(
        clt.n_layers,
        mode=str(lsw_cfg.get("mode", "uniform")),
        min_weight=float(lsw_cfg.get("min_weight", 1.0)),
        max_weight=float(lsw_cfg.get("max_weight", 4.0)),
    )
    if lsw_cfg.get("mode", "uniform") != "uniform":
        print(
            f"[train_clt] Per-layer sparsity weights ({lsw_cfg.get('mode')}): "
            f"L0={layer_weights[0]:.2f} .. L{clt.n_layers-1}={layer_weights[-1]:.2f}"
        )

    metrics_path = output_dir / "metrics.jsonl"
    micro_step = 0
    optimizer_step = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=max_optimizer_steps, desc="train CLT optimizer steps")
    while optimizer_step < max_optimizer_steps:
        for batch in iter_token_batches(cfg, tokenizer, device=device):
            if max_micro_steps is not None and micro_step >= max_micro_steps:
                break

            with torch.no_grad():
                acts = collector.run(batch.input_ids, batch.attention_mask)

            features, recons = clt(
                acts.mlp_inputs,
                mlp_targets=acts.mlp_outputs,
                update_normalization_stats=True,
            )
            rec_loss = reconstruction_loss(recons, acts.mlp_outputs)
            sp_loss = tanh_sparsity_loss(features, clt, c=sparsity_c, layer_weights=layer_weights)
            lambda_sparsity = get_lambda_sparsity(
                optimizer_step,
                target_lambda=lambda_sparsity_target,
                warmup_steps=sparsity_warmup_steps,
            )
            loss = rec_loss + lambda_sparsity * sp_loss

            (loss / grad_accum).backward()

            micro_step += 1

            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(clt.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                pbar.update(1)

                current_lr = optimizer.param_groups[0]["lr"]

                if optimizer_step == 1 or optimizer_step % log_every == 0:
                    summary = summarize_metrics(features, recons, acts.mlp_outputs)
                    row = {
                        "kind": "train",
                        "step": optimizer_step,
                        "optimizer_step": optimizer_step,
                        "micro_step": micro_step,
                        "lr": current_lr,
                        "lambda_sparsity": lambda_sparsity,
                        "loss": float(loss.item()),
                        "reconstruction_loss": float(rec_loss.item()),
                        "sparsity_loss": float(sp_loss.item()),
                        **summary,
                    }
                    with metrics_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    tqdm.write(
                        f"optimizer_step={optimizer_step} micro_step={micro_step} "
                        f"lr={current_lr:.2e} λ_sp={lambda_sparsity:.2e} "
                        f"loss={row['loss']:.4f} "
                        f"nmse={row['nmse_mean']:.4f} "
                        f"l0={row['l0_mean']:.2f}"
                    )

                # Eval on held-out split
                if eval_every > 0 and optimizer_step % eval_every == 0:
                    eval_metrics = eval_clt(
                        clt, collector, cfg, tokenizer, device
                    )
                    if eval_metrics:
                        eval_row = {
                            "kind": "eval",
                            "step": optimizer_step,
                            "optimizer_step": optimizer_step,
                            **eval_metrics,
                        }
                        with metrics_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(eval_row, ensure_ascii=False) + "\n")
                        tqdm.write(
                            f"[eval] step={optimizer_step} "
                            f"eval_nmse={eval_metrics['eval_nmse_mean']:.4f} "
                            f"eval_l0={eval_metrics['eval_l0_mean']:.2f}"
                        )

                if optimizer_step > 0 and optimizer_step % save_every == 0:
                    ckpt_path = output_dir / f"clt_step_{optimizer_step}.pt"
                    save_checkpoint(
                        {
                            "cfg": cfg,
                            "model_state_dict": clt.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict()
                            if scheduler is not None
                            else None,
                            "step": optimizer_step,
                            "optimizer_step": optimizer_step,
                            "micro_step": micro_step,
                            "init_from_checkpoint": str(init_checkpoint_path)
                            if init_checkpoint_path
                            else None,
                            "init_checkpoint_step": init_checkpoint.get("step")
                            if init_checkpoint
                            else None,
                        },
                        ckpt_path,
                    )
                    prune_old_step_checkpoints(output_dir, keep_last_checkpoints)

            if optimizer_step >= max_optimizer_steps:
                break

        if max_micro_steps is not None and micro_step >= max_micro_steps:
            break

    pbar.close()
    final_path = output_dir / "clt_final.pt"
    save_checkpoint(
        {
            "cfg": cfg,
            "model_state_dict": clt.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
            if scheduler is not None
            else None,
            "step": optimizer_step,
            "optimizer_step": optimizer_step,
            "micro_step": micro_step,
            "init_from_checkpoint": str(init_checkpoint_path)
            if init_checkpoint_path
            else None,
            "init_checkpoint_step": init_checkpoint.get("step")
            if init_checkpoint
            else None,
        },
        final_path,
    )
    return final_path
