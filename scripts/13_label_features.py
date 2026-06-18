#!/usr/bin/env python
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import re
from collections import Counter
from pathlib import Path

import _path_setup  # noqa: F401

import torch
from tqdm import tqdm

from qwen_clt.utils.config import load_config
from qwen_clt.models.qwen_hooks import (
    load_qwen_model_and_tokenizer,
    QwenMLPHookCollector,
)
from qwen_clt.replacement.loader import load_autoencoder_from_checkpoint
from qwen_clt.data.text_dataset import iter_token_batches

FEATURE_ID_RE = re.compile(r"L(\d+):P\d+:F(\d+)")


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
    data = json.loads(Path(path).read_text())
    found = set()
    for n in data.get("nodes", []):
        if n.get("type") == "CLTFeatureNode" and n.get("feature_idx") is not None:
            found.add((int(n["layer"]), int(n["feature_idx"])))
    return found


def main():
    ap = argparse.ArgumentParser(
        description="Label CLT features by their top-activating tokens."
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--feature", action="append", type=parse_feature_flag, default=[])
    ap.add_argument("--from-summary", default=None)
    ap.add_argument("--from-graph", default=None)
    ap.add_argument("--max-tokens", type=int, default=100000)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--context", type=int, default=6)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    feats = set(args.feature)
    if args.from_summary:
        feats |= features_from_summary(args.from_summary)
    if args.from_graph:
        feats |= features_from_graph(args.from_graph)
    if not feats:
        ap.error("no features selected; use --feature, --from-summary or --from-graph")
    feats = sorted(feats)
    print(f"labeling {len(feats)} features over up to {args.max_tokens} tokens")

    cfg = load_config(args.config)
    device = cfg["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_qwen_model_and_tokenizer(cfg)
    clt = load_autoencoder_from_checkpoint(args.checkpoint, device=device)
    collector = QwenMLPHookCollector(model)

    heaps = {lf: [] for lf in feats}
    tie = itertools.count()
    seen = 0

    pbar = tqdm(total=args.max_tokens, desc="scanning", unit="tok")
    for batch in iter_token_batches(cfg, tokenizer, device=device, rank=0, world_size=1):
        acts = collector.run(batch.input_ids, batch.attention_mask)
        features, _ = clt(acts.mlp_inputs)
        ids = batch.input_ids.detach().cpu()
        seq_len = ids.shape[1]
        for (layer, fidx) in feats:
            act = features[layer][:, :, fidx].float().detach().cpu().reshape(-1)
            k = min(args.top_k, act.numel())
            vals, idxs = torch.topk(act, k)
            heap = heaps[(layer, fidx)]
            for v, flat in zip(vals.tolist(), idxs.tolist()):
                if v <= 0:
                    continue
                if len(heap) >= args.top_k and v <= heap[0][0]:
                    continue
                b, p = divmod(flat, seq_len)
                ctx = ids[b, max(0, p - args.context): p + 1].tolist()
                item = (v, next(tie), ctx)
                if len(heap) < args.top_k:
                    heapq.heappush(heap, item)
                else:
                    heapq.heapreplace(heap, item)
        n = int(batch.input_ids.numel())
        seen += n
        pbar.update(n)
        if seen >= args.max_tokens:
            break
    pbar.close()

    result = {}
    for (layer, fidx) in feats:
        ordered = sorted(heaps[(layer, fidx)], key=lambda x: -x[0])
        examples = []
        tok_counts = Counter()
        for v, _, ctx in ordered:
            token = tokenizer.decode([ctx[-1]])
            context = tokenizer.decode(ctx)
            tok_counts[token.strip()] += 1
            examples.append({"activation": round(v, 4), "token": token, "context": context})
        label = ", ".join(t for t, _ in tok_counts.most_common(4) if t)
        result[f"L{layer}:F{fidx}"] = {
            "layer": layer,
            "feature": fidx,
            "label": label,
            "top_tokens": [t for t, _ in tok_counts.most_common(8)],
            "examples": examples,
        }
        print(f"L{layer}:F{fidx}  ->  {label or '(no positive activations)'}")

    out = args.output or "feature_labels.json"
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
