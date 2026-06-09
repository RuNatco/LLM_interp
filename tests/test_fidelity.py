import json

from qwen_clt.attribution.fidelity import (
    default_replacement_metrics_path,
    replacement_fidelity_report,
    summarize_replacement_fidelity,
)


def test_default_replacement_metrics_path_is_checkpoint_sibling():
    assert str(
        default_replacement_metrics_path(
            "outputs/base_clt_recon_fidelity_v2_continue_v2/clt_final.pt"
        )
    ) == (
        "outputs/base_clt_recon_fidelity_v2_continue_v2/"
        "replacement_eval_metrics.json"
    )


def test_fidelity_report_warns_on_low_last_token_top1():
    payload = {
        "metrics": {
            "last_token_top1_agreement": 0.1,
            "target_logit_diff_mae": 3.0,
        }
    }

    report = summarize_replacement_fidelity(
        payload,
        metrics_path="metrics.json",
        min_last_token_top1=0.3,
        max_target_logit_diff_mae=2.0,
    )

    assert report["passed"] is False
    assert len(report["warnings"]) == 2


def test_fidelity_report_loads_metrics_file(tmp_path):
    path = tmp_path / "replacement_eval_metrics.json"
    path.write_text(
        json.dumps({"metrics": {"last_token_top1_agreement": 0.5}}),
        encoding="utf-8",
    )

    report = replacement_fidelity_report(
        checkpoint_path=tmp_path / "clt_final.pt",
        metrics_path=path,
    )

    assert report["available"] is True
    assert report["passed"] is True
