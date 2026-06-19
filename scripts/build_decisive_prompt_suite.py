#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _path_setup  # noqa: F401

import torch

from qwen_clt.utils.config import load_config
from qwen_clt.models.qwen_hooks import load_qwen_model_and_tokenizer

PROMPTS = [
    ("Demand rises sharply while supply stays flat, so the price will", "rise"),
    ("Demand is greater than generation, so the price will", "rise"),
    ("A drought destroys much of the harvest, so the price will", "rise"),
    ("A product becomes scarce while buyers still need it, so the price will", "rise"),
    ("Fuel supply is cut while demand keeps growing, so the price will", "rise"),
    ("Buyers rush to stockpile a limited good, so the price will", "rise"),
    ("Shortages spread across the market, so the price will", "rise"),
    ("Production halts but orders keep coming, so the price will", "rise"),
    ("A new tariff raises import costs, so the price will", "rise"),
    ("Demand doubles overnight while stock is fixed, so the price will", "rise"),
    ("Supply chains break down during peak demand, so the price will", "rise"),
    ("Inventory runs low as customers keep buying, so the price will", "rise"),
    ("A bumper harvest floods the market with grain, so the price will", "fall"),
    ("Generation exceeds demand, so the price will", "fall"),
    ("Supply is far greater than demand, so the price will", "fall"),
    ("Factories double output while sales stay flat, so the price will", "fall"),
    ("Warehouses overflow with unsold goods, so the price will", "fall"),
    ("Demand collapses while supply stays high, so the price will", "fall"),
    ("A new competitor floods the market with cheap units, so the price will", "fall"),
    ("Buyers lose interest as stock piles up, so the price will", "fall"),
    ("Overproduction creates a large surplus, so the price will", "fall"),
    ("Many sellers compete for few buyers, so the price will", "fall"),
    ("Output rises sharply while demand shrinks, so the price will", "fall"),
    ("A subsidy makes the good far cheaper to produce, so the price will", "fall"),
    ("Demand is lower than supply, so the price will", "fall"),
    ("Consumers cut spending during a recession, so the price will", "fall"),
    ("A mine discovers a huge new deposit, so the price will", "fall"),
    ("Exports are banned and goods pile up at home, so the price will", "fall"),
    ("A festival drives a surge of buyers, so the price will", "rise"),
    ("Energy costs spike for every producer, so the price will", "rise"),
    ("A glut of supply meets weak demand, so the price will", "fall"),
    ("Scarcity worsens as the season ends, so the price will", "rise"),
]


def first_id(tok, text):
    return tok.encode(text, add_special_tokens=False)[0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(
        description="Generate economic price-direction prompts and keep only the "
        "ones the model is decisive about (|logit(rise)-logit(fall)| >= threshold), "
        "so the faithfulness metric is well-conditioned.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--up", default=" rise")
    ap.add_argument("--down", default=" fall")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--output", default="decisive_prompts.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = cfg["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_qwen_model_and_tokenizer(cfg)
    model.eval()
    up_id = first_id(tok, args.up)
    dn_id = first_id(tok, args.down)

    kept, dropped = [], []
    n_up = n_dn = 0
    for prompt, expected in PROMPTS:
        enc = tok(prompt, return_tensors="pt").to(device)
        lg = model(**enc).logits[0, -1].float()
        diff = float(lg[up_id] - lg[dn_id])
        decisive = abs(diff) >= args.threshold
        direction = "rise" if diff > 0 else "fall"
        agree = "OK" if direction == expected else "vs-expected"
        line = f"  [{'keep' if decisive else 'drop'}] diff={diff:+.2f} pred={direction:<4} exp={expected:<4} [{agree}]  {prompt}"
        print(line)
        if decisive:
            kept.append(prompt)
            n_up += direction == "rise"
            n_dn += direction == "fall"
        else:
            dropped.append(prompt)

    Path(args.output).write_text(json.dumps(kept, ensure_ascii=False, indent=2))
    print(f"\nkept {len(kept)} decisive / dropped {len(dropped)} borderline "
          f"(|diff| < {args.threshold})")
    print(f"direction balance among kept: rise={n_up}  fall={n_dn}")
    print(f"wrote {args.output}  (feed to scripts/09 via --prompts-file)")


if __name__ == "__main__":
    main()
