from __future__ import annotations

import argparse

import _path_setup  # noqa: F401

from qwen_clt.utils.config import load_config
from qwen_clt.training.train_clt import train_clt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ckpt = train_clt(cfg)
    print(f"Saved final CLT checkpoint to: {ckpt}")


if __name__ == "__main__":
    main()
