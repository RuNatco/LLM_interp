# Final Training Pipeline

Этот файл фиксирует итоговый маршрут с лучшими результатами. Исторический
экспериментальный перебор оставлен в ветке `main`; эта ветка содержит только
финальный pipeline.

Текущий основной checkpoint:

```text
outputs/base_clt_recon_fidelity_v2_continue_v2/clt_final.pt
```

Текущий основной config:

```text
configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v2.yaml
```

Лучшие replacement metrics:

```text
top1_agreement: 0.45196688025705
last_token_top1_agreement: 0.5366795366795367
kl_div: 1.341879009816182
last_token_kl_div: 1.2839431884666208
mean_abs_logit_diff: 1.3351546214621859
last_token_mean_abs_logit_diff: 1.3113203382400012
target_logit_diff_mae: 0.5119109504701548
```

Лучшие Deep Trace suite metrics:

```text
replacement fidelity gate: passed
mean sign_match: about 0.943
median sign_match: 1.000
mean MLP error norm: about 6.71
```

## 0. Project Root

```bash
cd /home/jupyter/project/LLM_interp
```

Use local project code first. Do not prepend old external project paths to
`PYTHONPATH`.

```bash
export PYTHONPATH=/home/jupyter/project/LLM_interp/src
```

Optional cache location:

```bash
export HF_HOME=/home/jupyter/project/LLM_interp/.cache/huggingface
```

## 1. Backup Intermediate Outputs

If the final checkpoint already exists and you only want to clean the cloud
workspace, run the backup now. If you are retraining from scratch, run this
after Stage 3, eval, and Deep Trace succeed.

Dry run:

```bash
bash scripts/10_backup_intermediate_outputs.sh
```

Apply backup:

```bash
bash scripts/10_backup_intermediate_outputs.sh --apply
```

This keeps the final output in place:

```text
outputs/base_clt_recon_fidelity_v2_continue_v2
```

and moves intermediate output dirs to:

```text
outputs/_backup/<timestamp>/
```

## 2. One-Command Final Pipeline

For a full final run with cleanup of intermediate outputs after success:

```bash
bash scripts/11_run_final_training_pipeline.sh --backup-intermediate
```

If the best checkpoint is already trained and you only need final eval, Deep
Trace, and backup:

```bash
bash scripts/11_run_final_training_pipeline.sh --skip-train --backup-intermediate
```

The script refuses to train into existing output dirs by default, because
training metrics are appended and checkpoints can be overwritten. If this is
intentional, pass:

```bash
bash scripts/11_run_final_training_pipeline.sh --allow-existing
```

## 3. Train Best Pipeline From Scratch

Run this only if the best checkpoints do not already exist.

Stage 1: train the 2048-feature reconstruction baseline.

```bash
python3 scripts/01_train_clt.py \
  --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2.yaml
```

Stage 2: continue from `recon_fidelity_v2`.

```bash
python3 scripts/01_train_clt.py \
  --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v1.yaml
```

Stage 3: final short continuation from `continue_v1`.

```bash
python3 scripts/01_train_clt.py \
  --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v2.yaml
```

## 4. Evaluate Final Replacement Model

```bash
python3 scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v2.yaml
```

Expected gate:

```text
last_token_top1_agreement >= 0.50
kl_div <= 1.40
last_token_kl_div <= 1.35
target_logit_diff_mae <= 0.55
```

Inspect metrics:

```bash
cat outputs/base_clt_recon_fidelity_v2_continue_v2/replacement_eval_metrics.json
```

## 5. Build Final Deep Trace Suite

```bash
python3 scripts/09_build_deep_trace_prompt_suite.py \
  --checkpoint outputs/base_clt_recon_fidelity_v2_continue_v2/clt_final.pt \
  --output-dir outputs/base_clt_recon_fidelity_v2_continue_v2/deep_trace_suite \
  --summary-output outputs/base_clt_recon_fidelity_v2_continue_v2/deep_trace_suite_summary.json \
  --max-feature-nodes 64 \
  --top-error-nodes 8 \
  --causal-top-k 8
```

Inspect summary:

```bash
cat outputs/base_clt_recon_fidelity_v2_continue_v2/deep_trace_suite_summary.json
```

Expected suite quality:

```text
replacement fidelity gate: passed
mean sign_match >= 0.90
median sign_match = 1.00
mean MLP error norm <= 7.00
```

## 6. Final Artifacts

Keep these as the final outputs:

```text
outputs/base_clt_recon_fidelity_v2_continue_v2/clt_final.pt
outputs/base_clt_recon_fidelity_v2_continue_v2/replacement_eval_metrics.json
outputs/base_clt_recon_fidelity_v2_continue_v2/deep_trace_suite_summary.json
outputs/base_clt_recon_fidelity_v2_continue_v2/deep_trace_suite/
```

All other experiment output dirs are intermediate and can remain under:

```text
outputs/_backup/<timestamp>/
```
