from __future__ import annotations

from pathlib import Path
import json
import torch
from tqdm import tqdm

from qwen_clt.data.text_dataset import iter_token_batches
from qwen_clt.models.qwen_hooks import load_qwen_model_and_tokenizer, QwenMLPHookCollector
from qwen_clt.models.cross_layer_transcoder import CrossLayerTranscoder
from qwen_clt.training.losses import reconstruction_loss, tanh_sparsity_loss
from qwen_clt.training.metrics import summarize_metrics
from qwen_clt.utils.seed import set_seed
from qwen_clt.utils.io import save_checkpoint


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
    ).to(device)

    optimizer = torch.optim.AdamW(
        clt.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )

    grad_accum = int(cfg["training"].get("gradient_accumulation_steps", 1))
    max_steps = int(cfg["training"]["max_steps"])
    log_every = int(cfg["training"].get("log_every", 50))
    save_every = int(cfg["training"].get("save_every", 1000))
    lambda_sparsity = float(cfg["training"].get("lambda_sparsity", 1e-4))
    sparsity_c = float(cfg["training"].get("sparsity_c", 1.0))
    grad_clip = float(cfg["training"].get("grad_clip_norm", 1.0))

    metrics_path = output_dir / "metrics.jsonl"
    step = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=max_steps, desc="train CLT")
    while step < max_steps:
        for batch in iter_token_batches(cfg, tokenizer, device=device):
            with torch.no_grad():
                acts = collector.run(batch.input_ids, batch.attention_mask)

            features, recons = clt(acts.mlp_inputs)
            rec_loss = reconstruction_loss(recons, acts.mlp_outputs)
            sp_loss = tanh_sparsity_loss(features, clt, c=sparsity_c)
            loss = rec_loss + lambda_sparsity * sp_loss
            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(clt.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % log_every == 0:
                summary = summarize_metrics(features, recons, acts.mlp_outputs)
                row = {
                    "step": step,
                    "loss": float(loss.item()),
                    "reconstruction_loss": float(rec_loss.item()),
                    "sparsity_loss": float(sp_loss.item()),
                    **summary,
                }
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                tqdm.write(f"step={step} loss={row['loss']:.4f} nmse={row['nmse_mean']:.4f} l0={row['l0_mean']:.2f}")

            if step > 0 and step % save_every == 0:
                ckpt_path = output_dir / f"clt_step_{step}.pt"
                save_checkpoint({"cfg": cfg, "model_state_dict": clt.state_dict(), "step": step}, ckpt_path)

            step += 1
            pbar.update(1)
            if step >= max_steps:
                break

    pbar.close()
    final_path = output_dir / "clt_final.pt"
    save_checkpoint({"cfg": cfg, "model_state_dict": clt.state_dict(), "step": step}, final_path)
    return final_path
