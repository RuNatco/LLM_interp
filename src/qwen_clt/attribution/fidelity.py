from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_replacement_metrics_path(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path).parent / "replacement_eval_metrics.json"


def load_replacement_eval_payload(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise TypeError(f"Expected replacement metrics JSON object, got {type(payload)}")

    return payload


def summarize_replacement_fidelity(
    payload: dict[str, Any],
    *,
    metrics_path: str | Path,
    min_last_token_top1: float = 0.3,
    max_target_logit_diff_mae: float | None = 2.0,
) -> dict[str, Any]:
    metrics = payload.get("metrics", payload)
    if not isinstance(metrics, dict):
        raise TypeError("Replacement eval payload must contain a 'metrics' object.")

    last_top1 = metrics.get("last_token_top1_agreement")
    if last_top1 is None:
        last_top1 = metrics.get("top1_agreement")

    target_mae = metrics.get("target_logit_diff_mae")
    warnings: list[str] = []

    if last_top1 is None:
        warnings.append(
            "No last-token top1 metric found; cannot judge replacement fidelity."
        )
    elif float(last_top1) < min_last_token_top1:
        warnings.append(
            "Low replacement fidelity: "
            f"last_token_top1_agreement={float(last_top1):.4f} "
            f"< {min_last_token_top1:.4f}."
        )

    if (
        max_target_logit_diff_mae is not None
        and target_mae is not None
        and float(target_mae) > max_target_logit_diff_mae
    ):
        warnings.append(
            "Poor target-direction fidelity: "
            f"target_logit_diff_mae={float(target_mae):.4f} "
            f"> {max_target_logit_diff_mae:.4f}."
        )

    selected_metrics = {
        key: metrics[key]
        for key in [
            "top1_agreement",
            "last_token_top1_agreement",
            "kl_div",
            "last_token_kl_div",
            "mean_abs_logit_diff",
            "last_token_mean_abs_logit_diff",
            "target_logit_diff_mae",
        ]
        if key in metrics
    }

    return {
        "available": True,
        "metrics_path": str(Path(metrics_path)),
        "passed": len(warnings) == 0,
        "thresholds": {
            "min_last_token_top1": float(min_last_token_top1),
            "max_target_logit_diff_mae": max_target_logit_diff_mae,
        },
        "metrics": selected_metrics,
        "warnings": warnings,
    }


def replacement_fidelity_report(
    *,
    checkpoint_path: str | Path,
    metrics_path: str | Path | None,
    min_last_token_top1: float = 0.3,
    max_target_logit_diff_mae: float | None = 2.0,
) -> dict[str, Any]:
    path = Path(metrics_path) if metrics_path else default_replacement_metrics_path(checkpoint_path)

    if not path.exists():
        return {
            "available": False,
            "metrics_path": str(path),
            "passed": False,
            "thresholds": {
                "min_last_token_top1": float(min_last_token_top1),
                "max_target_logit_diff_mae": max_target_logit_diff_mae,
            },
            "metrics": {},
            "warnings": [
                "Replacement eval metrics not found. Run "
                "scripts/02_eval_replacement_model.py before interpreting graph results."
            ],
        }

    payload = load_replacement_eval_payload(path)
    return summarize_replacement_fidelity(
        payload,
        metrics_path=path,
        min_last_token_top1=min_last_token_top1,
        max_target_logit_diff_mae=max_target_logit_diff_mae,
    )


def print_fidelity_report(
    report: dict[str, Any],
    *,
    strict: bool = False,
) -> None:
    print(f"Replacement metrics: {report['metrics_path']}")

    for key, value in report.get("metrics", {}).items():
        print(f"  {key}: {value}")

    warnings = report.get("warnings", [])
    if warnings:
        for warning in warnings:
            print(f"[fidelity warning] {warning}")
        if strict:
            raise RuntimeError(
                "Replacement fidelity gate failed. Re-run with better CLT fidelity "
                "or omit --strict-fidelity to keep this as a diagnostic proxy run."
            )
    else:
        print("Replacement fidelity gate: passed")
