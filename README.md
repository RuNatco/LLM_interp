# Qwen CLT Final Pipeline

Чистая ветка с итоговым пайплайном обучения и анализа
**Qwen/Qwen2.5-0.5B + Cross-Layer Transcoder (CLT)**.

Исторические эксперименты, старые configs и промежуточные графы оставлены в
`main`. Эта ветка содержит только финальный воспроизводимый маршрут:

```text
train CLT reconstruction baseline
-> continue from checkpoint
-> final low-LR continuation
-> replacement model evaluation
-> Deep Trace stage 1 prompt suite
```

## Лучший checkpoint

Финальный checkpoint после облачного запуска:

```text
outputs/base_clt_recon_fidelity_v2_continue_v2/clt_final.pt
```

Финальный config:

```text
configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v2.yaml
```

Лучшие полученные replacement metrics:

```text
top1_agreement: 0.45196688025705
last_token_top1_agreement: 0.5366795366795367
kl_div: 1.341879009816182
last_token_kl_div: 1.2839431884666208
target_logit_diff_mae: 0.5119109504701548
```

Лучший Deep Trace prompt suite:

```text
replacement fidelity gate: passed
mean sign_match: about 0.943
median sign_match: 1.000
mean MLP error norm: about 6.71
```

## Структура

```text
FINAL_TRAINING_PIPELINE.md

configs/
  qwen2_5_0_5b_base_clt_recon_fidelity_v2.yaml
  qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v1.yaml
  qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v2.yaml

scripts/
  01_train_clt.py
  02_eval_replacement_model.py
  09_build_deep_trace_prompt_suite.py
  10_backup_intermediate_outputs.sh
  11_run_final_training_pipeline.sh

src/qwen_clt/
  data/
  models/
  replacement/
  training/
  deep_trace/
  interventions/
  attribution/      # minimal target/fidelity/validation helpers for Deep Trace
  utils/
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt
pip install -e .
```

В облаке запускай из корня проекта:

```bash
cd /home/jupyter/project/LLM_interp
export PYTHONPATH=/home/jupyter/project/LLM_interp/src
```

## Один итоговый запуск

Если checkpoint уже обучен и нужно только пересчитать final eval, Deep Trace и
сложить промежуточные outputs в backup:

```bash
bash scripts/11_run_final_training_pipeline.sh --skip-train --backup-intermediate
```

Если нужно воспроизвести финальный пайплайн с нуля:

```bash
bash scripts/11_run_final_training_pipeline.sh --backup-intermediate
```

Скрипт по умолчанию откажется обучать в уже существующие output dirs, чтобы не
перезаписать checkpoint и не смешать `metrics.jsonl`. Если повторный запуск в
эти же папки нужен осознанно:

```bash
bash scripts/11_run_final_training_pipeline.sh --allow-existing
```

Подробные команды и expected gates описаны в
`FINAL_TRAINING_PIPELINE.md`.

## Backup промежуточных результатов

Dry run:

```bash
bash scripts/10_backup_intermediate_outputs.sh
```

Реальный перенос:

```bash
bash scripts/10_backup_intermediate_outputs.sh --apply
```

Скрипт сохраняет финальную папку на месте:

```text
outputs/base_clt_recon_fidelity_v2_continue_v2
```

а промежуточные output dirs переносит в:

```text
outputs/_backup/<timestamp>/
```

## Ограничение метода

Deep Trace stage 1 глубже proxy attribution, потому что включает replacement
cache, residual/layernorm/attention/error nodes и causal feature ablations.
Но это ещё не full path attribution: attention представлен layer-level output,
а feature-to-residual edges остаются decoder-write proxies.
