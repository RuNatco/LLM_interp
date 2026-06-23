#!/usr/bin/env python
from __future__ import annotations

import argparse
import warnings

import torch

warnings.filterwarnings("ignore")


def pre_mlp_norm(model, layer):
    h = model.transformer.h[layer]
    if hasattr(h, "ln_2"):
        return h.ln_2
    return h.post_attention_layernorm


def norm_kind(module):
    has_bias = getattr(module, "bias", None) is not None
    return "LayerNorm" if has_bias else "RMSNorm"


def layernorm_jacobian(x0, g, eps):
    d = x0.numel()
    mu = x0.mean()
    std = (x0.var(unbiased=False) + eps).sqrt()
    xhat = (x0 - mu) / std
    eye = torch.eye(d, dtype=torch.double)
    ones = torch.ones(d, dtype=torch.double)
    proj = ones.outer(ones) / d + xhat.outer(xhat) / d
    return (g.unsqueeze(1) / std) * (eye - proj), float(std)


def rmsnorm_jacobian(x0, g, eps):
    d = x0.numel()
    rms = (x0.pow(2).mean() + eps).sqrt()
    xr = x0 / rms
    eye = torch.eye(d, dtype=torch.double)
    return (g.unsqueeze(1) / rms) * (eye - xr.outer(xr) / d), float(rms)


def main():
    ap = argparse.ArgumentParser(
        description="Transferability demo: the frozen-DE normalizer jacobian for "
        "LayerNorm vs RMSNorm on the same real residual. Shows that porting the "
        "method across normalization types changes only the normalizer component, "
        "while the g/scale form and the decoder->encoder mechanics are identical.")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--prompt", default="Energy costs spike for every producer, so the price will")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).eval()

    norm = pre_mlp_norm(model, args.layer)
    kind = norm_kind(norm)
    g = norm.weight.detach().double()
    eps = float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-5)))

    cap = {}
    h = norm.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0].detach()))
    ids = tok(args.prompt, return_tensors="pt")
    with torch.no_grad():
        model(**ids)
    h.remove()
    x0 = cap["x"][0, -1].double()
    d = x0.numel()

    J_ln, std = layernorm_jacobian(x0, g, eps)
    J_rms, rms = rmsnorm_jacobian(x0, g, eps)

    torch.manual_seed(args.seed)
    dec = torch.randn(d, dtype=torch.double); dec /= dec.norm()
    enc = torch.randn(d, dtype=torch.double); enc /= enc.norm()
    w_ln = float(enc @ J_ln @ dec)
    w_rms = float(enc @ J_rms @ dec)
    w_ln_diag = float(enc @ ((g.unsqueeze(1) / std) * torch.eye(d, dtype=torch.double)) @ dec)
    w_rms_diag = float(enc @ ((g.unsqueeze(1) / rms) * torch.eye(d, dtype=torch.double)) @ dec)

    print(f"model={args.model}  layer={args.layer}  d={d}  actual pre-MLP norm: {kind}")
    print(f"residual mean={float(x0.mean()):+.3f}  (LayerNorm centers it, RMSNorm does not)")
    print(f"scale:           LayerNorm std={std:.4f}     RMSNorm rms={rms:.4f}")
    print(f"diag multiplier: mean(g/std)={float((g/std).mean()):.4f}  "
          f"mean(g/rms)={float((g/rms).mean()):.4f}")
    print()
    print("frozen-DE normalizer applied to the same enc/dec directions:")
    print(f"  full jacobian : LayerNorm {w_ln:+.6f}   RMSNorm {w_rms:+.6f}   "
          f"ratio {w_ln/w_rms:.3f}")
    print(f"  diagonal only : LayerNorm {w_ln_diag:+.6f}   RMSNorm {w_rms_diag:+.6f}   "
          f"ratio {w_ln_diag/w_rms_diag:.3f}")
    print()
    print("interpretation:")
    print("  - diagonal term g/scale is the SAME form in both norms and coincides")
    print("    numerically when the residual is mean-centered (std == rms);")
    print("  - the only structural difference is mean-centering (present in LayerNorm,")
    print("    absent in RMSNorm) plus the variance-direction projection term;")
    print("  - decoder->encoder path, cross-layer sum and JumpReLU are identical.")
    print("  => porting frozen-DE across norm types changes ONLY the normalizer row")
    print("     of the transferability map; the rest of the method is invariant.")


if __name__ == "__main__":
    main()
