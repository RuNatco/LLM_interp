#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _path_setup  # noqa: F401

import torch

from qwen_clt.utils.config import load_config
from qwen_clt.models.qwen_hooks import load_qwen_model_and_tokenizer
from qwen_clt.replacement.loader import load_autoencoder_from_checkpoint

FEATURE_ID_RE = re.compile(r"L(\d+):P\d+:F(\d+)")
LATIN = re.compile(r"[A-Za-z][A-Za-z\-]+")
ASCII_OK = re.compile(r"[\x21-\x7e]+")


def parse_feature_flag(value):
    m = re.fullmatch(r"L?(\d+):F?(\d+)", value.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"bad --feature {value!r}, want LAYER:FEATURE")
    return (int(m.group(1)), int(m.group(2)))


def features_from_summary(path):
    data = json.loads(Path(path).read_text())
    found = set()
    for r in data.get("results", []):
        for bucket in ("top_causal_edges", "top_causal_residual_delta_edges"):
            for e in r.get(bucket, []):
                for node in (e.get("source", ""), e.get("target", "")):
                    m = FEATURE_ID_RE.search(node)
                    if m:
                        found.add((int(m.group(1)), int(m.group(2))))
    return found


def features_from_graph(path):
    found = set()
    try:
        data = json.loads(Path(path).read_text())
    except Exception:
        return found
    for n in data.get("nodes", []):
        if n.get("type") == "CLTFeatureNode" and n.get("feature_idx") is not None:
            found.add((int(n["layer"]), int(n["feature_idx"])))
    return found


def collect_features(args):
    feats = set(args.feature)
    graph_files = list(args.from_graph or [])
    if args.graphs_dir:
        graph_files += sorted(str(p) for p in Path(args.graphs_dir).glob("*.json"))
    used = []
    for gp in graph_files:
        gf = features_from_graph(gp)
        if gf:
            used.append((gp, len(gf)))
        feats |= gf
    if args.from_summary:
        feats |= features_from_summary(args.from_summary)
    return feats, used


def first_token_id(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)["input_ids"][0]


def final_norm_weight(model):
    for path in ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm"):
        obj = model
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and hasattr(obj, "weight"):
            return obj.weight.detach()
    return None


def keep_token(tok, mode):
    s = tok.strip()
    if mode == "none":
        return True
    if mode == "ascii":
        return bool(ASCII_OK.fullmatch(s)) and len(s) >= 2
    return bool(LATIN.fullmatch(s)) and len(s) >= 2


def clean_list(tokenizer, idx, mode, k):
    out, seen = [], set()
    for i in idx.tolist():
        t = tokenizer.decode([i])
        if keep_token(t, mode) and t.strip().lower() not in seen:
            seen.add(t.strip().lower())
            out.append(t)
        if len(out) >= k:
            break
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Label CLT features by their DIRECT logit effect on the "
        "target metric (logit positive - logit negative). top_promoted uses a "
        "mean-centred cosine projection and a token filter to suppress the "
        "high-norm multilingual/code artefacts of the raw logit lens."
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--feature", action="append", type=parse_feature_flag, default=[])
    ap.add_argument("--from-graph", action="append", default=[])
    ap.add_argument("--graphs-dir", default=None)
    ap.add_argument("--from-summary", default=None)
    ap.add_argument("--positive", default=" increase")
    ap.add_argument("--negative", default=" decrease")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--candidates", type=int, default=150)
    ap.add_argument("--token-filter", choices=["latin", "ascii", "none"], default="latin")
    ap.add_argument("--min-effect-norm", type=float, default=0.9)
    ap.add_argument("--output", default="feature_effect_labels.json")
    args = ap.parse_args()

    feats, used = collect_features(args)
    if not feats:
        ap.error("no features selected; use --feature, --from-graph, --graphs-dir or --from-summary")
    feats = sorted(feats)
    for gp, c in used:
        print(f"  {Path(gp).name}: {c} features")
    print(f"labeling {len(feats)} unique features across {len(used)} graph(s)")

    cfg = load_config(args.config)
    device = cfg["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_qwen_model_and_tokenizer(cfg)
    clt = load_autoencoder_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    W_U = model.get_output_embeddings().weight.detach().to(device).float()
    W_U_c = W_U - W_U.mean(0, keepdim=True)
    tok_norm = W_U_c.norm(dim=1) + 1e-6
    norm_w = final_norm_weight(model)
    norm_w = (norm_w.to(device).float() if norm_w is not None
              else torch.ones(W_U.shape[1], device=device))

    pos_id = first_token_id(tokenizer, args.positive)
    neg_id = first_token_id(tokenizer, args.negative)
    diff_dir = W_U[pos_id] - W_U[neg_id]

    nc = args.candidates
    result = {}
    for (layer, fidx) in feats:
        with torch.no_grad():
            eff = torch.zeros(clt.d_model, device=device, dtype=torch.float32)
            for tgt in range(layer, clt.n_layers):
                eff += clt.decoder_row_for_raw_output(layer, tgt, fidx).to(device).float()
            scaled = eff * norm_w
            target_effect = float(scaled @ diff_dir)
            cos = (scaled @ W_U_c.t()) / tok_norm
            top_i = torch.topk(cos, nc).indices
            bot_i = torch.topk(-cos, nc).indices
            promoted = clean_list(tokenizer, top_i, args.token_filter, args.top_k)
            suppressed = clean_list(tokenizer, bot_i, args.token_filter, args.top_k)

        norm = float(eff.norm())
        pushes = "increase" if target_effect > 0 else "decrease"
        concept = promoted[0].strip() if (promoted and norm >= args.min_effect_norm) else "?"
        result[f"L{layer}:F{fidx}"] = {
            "layer": layer,
            "feature": fidx,
            "target_effect": round(target_effect, 4),
            "pushes": pushes,
            "effect_norm": round(norm, 4),
            "concept": concept,
            "top_promoted": promoted,
            "top_suppressed": suppressed,
            "label": f"\u2192 {pushes} ({target_effect:+.2f})"
                     + (f"; \u00ab{concept}\u00bb" if concept != "?" else ""),
        }
        print(f"L{layer}:F{fidx}  ->  {pushes} ({target_effect:+.3f})  \u00ab{concept}\u00bb  {promoted[:4]}")

    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
