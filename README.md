# Qwen CLT Pipeline

Обучение и анализ **Cross-Layer Transcoder (CLT)** поверх MLP-слоёв
**Qwen/Qwen2.5-0.5B** для механистической интерпретируемости.

CLT — разреженный автоэнкодер: энкодер на каждый слой читает вход MLP,
треугольная матрица декодеров `src→tgt` реконструирует выходы MLP всех
последующих слоёв. После обучения все 24 MLP-блока Qwen заменяются
реконструкциями CLT (replacement model), и полученное пространство фич
используется для построения каузальных Deep Trace графов.

Цель проекта — **воспроизвести этот interpretability-пайплайн и показать его
состоятельность** на одной небольшой модели, с возможностью запуска через
Docker на нескольких GPU.

Полный маршрут:

```text
train CLT (stage 1, с нуля)
-> continue from checkpoint (stage 2)
-> final low-LR continuation (stage 3)
-> replacement model evaluation + control baselines (gate)
-> Deep Trace stage 2 prompt suite (gate)
```

Детали стадий и ожидаемые gates: [FINAL_TRAINING_PIPELINE.md](FINAL_TRAINING_PIPELINE.md).
Запуск в Docker: [DOCKER.md](DOCKER.md).

---

## Быстрый старт

### Docker (рекомендуется)

Степень параллелизма задаётся `GPUS` (число процессов torchrun); отдельных
конфигов под каждое число GPU нет.

```bash
docker compose build

# полный пайплайн на N GPU: train (3 стадии) -> eval+baselines -> Deep Trace
GPUS=4 docker compose run --rm pipeline

# только тренировка одной стадии
GPUS=2 docker compose run --rm train

# eval (с контрольными бейзлайнами) и Deep Trace по готовому чекпоинту
docker compose run --rm eval
docker compose run --rm deep-trace
```

### Без Docker

```bash
python -m venv .venv && source .venv/bin/activate
# torch ставится отдельно под вашу CUDA:
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -c constraints.txt
pip install -e .
```

Единый лаунчер (сам выбирает python3 / torchrun по числу GPU):

```bash
# 1 GPU
bash scripts/run.sh --gpus 1 \
  --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2.yaml

# N GPU (DDP)
bash scripts/run.sh --gpus 4 \
  --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2.yaml

# весь пайплайн на N GPU одной командой
bash scripts/11_run_final_training_pipeline.sh --gpus 4 --backup-intermediate
```

---

## Пайплайн: 2048 фич, 3 стадии

| Стадия  | Config                                            | batch/GPU | lr    |
|---------|---------------------------------------------------|-----------|-------|
| stage 1 | `..._recon_fidelity_v2.yaml`                      | 16        | 2e-4  |
| stage 2 | `..._recon_fidelity_v2_continue_v1.yaml`          | 16        | —     |
| stage 3 | `..._recon_fidelity_v2_continue_v2.yaml`          | 16        | 5e-5  |

Батч в конфиге — **на один GPU**; эффективный батч = `batch_size_sequences × N`.
При большом N масштабируйте lr под свой эффективный батч.

Каждая следующая стадия инициализируется из `clt_final.pt` предыдущей через
`training.init_from_checkpoint`. Финальный чекпоинт:
`outputs/base_clt_recon_fidelity_v2_continue_v2/clt_final.pt`.

---

## Контроль качества (gates)

### Replacement eval (`replacement_eval_metrics.json`)

| Метрика | Порог |
|---|---|
| `last_token_top1_agreement` | ≥ 0.50 |
| `kl_div` | ≤ 1.40 |
| `last_token_kl_div` | ≤ 1.35 |
| `target_logit_diff_mae` | ≤ 0.55 |

Текущий результат финального чекпоинта:

```text
last_token_top1_agreement: 0.537
kl_div:                    1.342
last_token_kl_div:         1.284
target_logit_diff_mae:     0.512
```

**Контрольные бейзлайны.** Eval считает те же метрики для контролей
(`zero` / `mean` / `random_clt`) и кладёт таблицу `baseline_comparison` в JSON:
обученный CLT должен быть заметно ближе к оригиналу, чем зануление MLP,
mean-ablation и необученный CLT. Без этого разрыва абсолютные числа ничего не
доказывают. Включается блоком `replacement_eval.baselines` в конфиге или
флагом `--baselines all`.

### Deep Trace suite (`deep_trace_suite_summary.json`)

| Метрика | Порог |
|---|---|
| replacement fidelity gate | passed |
| mean sign_match | ≥ 0.90 |
| median sign_match | = 1.00 |
| mean MLP error norm | ≤ 7.00 |

Текущий результат: mean sign_match ≈ 0.943, median = 1.00.

---

## Структура проекта

```text
configs/
  qwen2_5_0_5b_base_clt_recon_fidelity_v2.yaml            # stage 1
  qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v1.yaml # stage 2
  qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v2.yaml # stage 3 (final)

scripts/
  run.sh                          # единый лаунчер: 1 GPU / N-GPU DDP / resume
  01_train_clt.py                 # обучение CLT
  02_eval_replacement_model.py    # replacement метрики + контрольные бейзлайны + gate
  09_build_deep_trace_prompt_suite.py  # каузальные Deep Trace графы
  10_backup_intermediate_outputs.sh    # перенос промежуточных outputs в _backup
  11_run_final_training_pipeline.sh    # оркестратор всего пайплайна (--gpus N)

src/qwen_clt/
  models/        # CrossLayerTranscoder, Qwen hooks, proxy replacement model
  training/      # train loop (DDP), losses, train-time метрики
  data/          # токенизация с кешем + DataLoader с DistributedSampler
  replacement/   # LayerReplacementHook, baselines (zero/mean/random), метрики, loader
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
- Логи, eval и чекпоинты пишет только rank 0
- Backend NCCL по умолчанию; при проблемах совместимости NCCL/CUDA:
  `export DIST_BACKEND=gloo`
- Возобновление: `--continue-from <ckpt>` в `run.sh` (передаётся через env
  `CLT_OVERRIDE_INIT_CHECKPOINT`, загружаются веса модели)

---

## Тесты

```bash
PYTHONPATH=src python3 -m pytest tests/ -q
```

Покрывают: формы тензоров CLT и нормализацию, replacement hook и метрики,
контрольные бейзлайны, fidelity gate, валидацию знаков аблаций, сериализацию
Deep Trace графов.

---

## Ограничение метода

Deep Trace stage 2 глубже proxy attribution: включает replacement cache,
residual/layernorm/error nodes, per-head attention nodes и каузальные feature
ablations. Attention разложен до уровня вкладов `o_proj` — это больше, чем
layer-level attention, но ещё не полный token-to-token attention path.
Для top-k фич рёбра feature→residual строятся по каузальным дельтам кеша;
для остальных фич остаются proxy-рёбра через веса декодера.
