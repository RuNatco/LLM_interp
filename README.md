# Qwen CLT Pipeline

Обучение и анализ **Cross-Layer Transcoder (CLT)** поверх MLP-слоёв
**Qwen/Qwen2.5-0.5B** для механистической интерпретируемости.

CLT — разреженный автоэнкодер: энкодер на каждый слой читает вход MLP,
треугольная матрица декодеров `src→tgt` реконструирует выходы MLP всех
последующих слоёв. После обучения все 24 MLP-блока Qwen заменяются
реконструкциями CLT, и полученное пространство фич используется для
построения каузальных Deep Trace графов.

Полный маршрут:

```text
train CLT (stage 1, с нуля)
-> continue from checkpoint (stage 2, low LR)
-> replacement model evaluation (gate)
-> Deep Trace stage 2 prompt suite (gate)
```

Подробное описание каждого шага: [PIPELINE_DESCRIPTION.md](PIPELINE_DESCRIPTION.md).
Запуск в Docker: [DOCKER.md](DOCKER.md).

---

## Быстрый старт

### Docker (рекомендуется)

```bash
docker compose build

# 1 GPU
docker compose run --rm train-1gpu

# 2 GPU (DDP)
docker compose run --rm train-2gpu

# eval после обучения
docker compose run --rm eval --config configs/qwen2_5_0_5b_4096f_v1_2gpu.yaml
```

### Без Docker

```bash
python -m venv .venv && source .venv/bin/activate
# torch ставится отдельно под вашу CUDA:
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -c constraints.txt
pip install -e .
```

Запуск через единый лаунчер (сам выбирает python3 / torchrun):

```bash
# 1 GPU
bash scripts/run.sh --gpus 1 --config configs/qwen2_5_0_5b_4096f_v1.yaml

# N GPU (DDP)
bash scripts/run.sh --gpus 2 --config configs/qwen2_5_0_5b_4096f_v1_2gpu.yaml
bash scripts/run.sh --gpus 4 --config configs/qwen2_5_0_5b_4096f_v1_4gpu.yaml

# продолжить с чекпоинта (переопределяет training.init_from_checkpoint)
bash scripts/run.sh --gpus 2 \
  --config configs/qwen2_5_0_5b_4096f_v1_2gpu.yaml \
  --continue-from outputs/4096f_v1_2gpu/clt_step_5000.pt
```

---

## Текущий эксперимент: 4096 фич

| Config | Режим | batch/GPU | seq_len | eff. batch | lr |
|---|---|---|---|---|---|
| `qwen2_5_0_5b_4096f_v1.yaml` | 1 GPU | 16 | 128 | 32 seq | 0.0002 |
| `qwen2_5_0_5b_4096f_v1_2gpu.yaml` | 2 GPU DDP | 32 | 256 | 64 seq | 0.0003 |
| `qwen2_5_0_5b_4096f_v1_4gpu.yaml` | 4 GPU DDP | — | — | — | — |
| `qwen2_5_0_5b_4096f_v1_continue.yaml` | 1 GPU finetune | 16 | 128 | 32 seq | 0.00002 |

Цепочка: stage 1 (30000 шагов с нуля) → stage 2 `*_continue` (5000 шагов,
lr ×0.1, инициализация из `outputs/4096f_v1/clt_final.pt`).

> **Диск:** чекпоинт 4096f весит ~6–8 GB (декодеры 300 пар × 4096×896 + Adam
> state). `save_every`/`keep_last_checkpoints` в конфигах подобраны так, чтобы
> не переполнить диск — перед запуском проверьте `df -h`.

### Полный цикл 4096f

```bash
# Stage 1
bash scripts/run.sh --gpus 2 --config configs/qwen2_5_0_5b_4096f_v1_2gpu.yaml

# Stage 2 (continuation)
bash scripts/run.sh --gpus 1 --config configs/qwen2_5_0_5b_4096f_v1_continue.yaml

# Replacement eval
python3 scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_4096f_v1_continue.yaml

# Deep Trace suite
python3 scripts/09_build_deep_trace_prompt_suite.py \
  --checkpoint outputs/4096f_v1_continue/clt_final.pt \
  --output-dir outputs/4096f_v1_continue/deep_trace_suite \
  --summary-output outputs/4096f_v1_continue/deep_trace_suite_summary.json \
  --max-feature-nodes 64 --top-error-nodes 8 --causal-top-k 8
```

---

## Контроль качества (gates)

### Replacement eval (`replacement_eval_metrics.json`)

| Метрика | Порог |
|---|---|
| `last_token_top1_agreement` | ≥ 0.50 |
| `kl_div` | ≤ 1.40 |
| `last_token_kl_div` | ≤ 1.35 |
| `target_logit_diff_mae` | ≤ 0.55 |

### Deep Trace suite (`deep_trace_suite_summary.json`)

| Метрика | Порог |
|---|---|
| replacement fidelity gate | passed |
| mean sign_match | ≥ 0.90 |
| median sign_match | = 1.00 |
| mean MLP error norm | ≤ 7.00 |

### Baseline для сравнения (2048 фич, предыдущее поколение)

```text
checkpoint: outputs/base_clt_recon_fidelity_v2_continue_v2/clt_final.pt
last_token_top1_agreement: 0.537
kl_div:                    1.342
last_token_kl_div:         1.284
target_logit_diff_mae:     0.512
mean sign_match:           ~0.943
```

Эксперимент 4096f считается успешным, если превосходит эти значения.

---

## Структура проекта

```text
configs/
  qwen2_5_0_5b_4096f_v1.yaml              # stage 1, 1 GPU
  qwen2_5_0_5b_4096f_v1_2gpu.yaml         # stage 1, 2 GPU DDP
  qwen2_5_0_5b_4096f_v1_4gpu.yaml         # stage 1, 4 GPU DDP
  qwen2_5_0_5b_4096f_v1_continue.yaml     # stage 2 finetune
  qwen2_5_0_5b_base_clt_recon_fidelity_v2*.yaml  # baseline 2048f (3 стадии)

scripts/
  run.sh                          # единый лаунчер: 1 GPU / N-GPU DDP / resume
  01_train_clt.py                 # обучение CLT
  02_eval_replacement_model.py    # replacement метрики + gate
  09_build_deep_trace_prompt_suite.py  # каузальные Deep Trace графы
  10_backup_intermediate_outputs.sh    # перенос промежуточных outputs в _backup
  11_run_final_training_pipeline.sh    # оркестратор baseline-пайплайна (2048f)
  12_train_multigpu.sh            # multi-GPU лаунчер + авто-eval

src/qwen_clt/
  models/        # CrossLayerTranscoder, Qwen hooks, proxy replacement model
  training/      # train loop (DDP), losses, train-time метрики
  data/          # токенизация с кешем + DataLoader с DistributedSampler
  replacement/   # LayerReplacementHook, replacement метрики, loader
  attribution/   # целевые векторы, fidelity gate, валидация знаков
  deep_trace/    # построение графов, кеш активаций, сериализация, сводки
  interventions/ # FeatureIntervention (каузальные аблации фич)
  utils/         # config, atomic checkpoint IO, seed

tests/           # юнит-тесты (pytest, GPU не требуется)
Dockerfile, docker-compose.yml, DOCKER.md
```

---

## Multi-GPU: как это работает

- Запуск через `torchrun`, инициализация по `LOCAL_RANK` (без torchrun —
  обычный 1-GPU режим, ничего настраивать не нужно)
- CLT оборачивается в `DistributedDataParallel`; замороженный Qwen на каждом
  ранке работает как обычный inference без DDP
- `DistributedSampler` делит датасет между ранками без пересечений
- `max_train_tokens` в конфиге — **per-rank**: 2 GPU суммарно видят 2×
- Логи, eval и чекпоинты пишет только rank 0
- Backend NCCL по умолчанию; при проблемах совместимости NCCL/CUDA:
  `export DIST_BACKEND=gloo`
- Возобновление: `--continue-from <ckpt>` в `run.sh` (передаётся через env
  `CLT_OVERRIDE_INIT_CHECKPOINT`, загружаются только веса модели);
  scheduler восстанавливается только при `training.restore_scheduler_state: true`

---

## Тесты

```bash
PYTHONPATH=src python3 -m pytest tests/ -q
```

Покрывают: формы тензоров CLT и нормализацию, replacement hook и метрики,
fidelity gate, валидацию знаков аблаций, сериализацию Deep Trace графов.

---

## Ограничение метода

Deep Trace stage 2 глубже proxy attribution: включает replacement cache,
residual/layernorm/error nodes, per-head attention nodes и каузальные feature
ablations. Attention разложен до уровня вкладов `o_proj` — это больше, чем
layer-level attention, но ещё не полный token-to-token attention path.
Для top-k фич рёбра feature→residual строятся по каузальным дельтам кеша;
для остальных фич остаются proxy-рёбра через веса декодера.
