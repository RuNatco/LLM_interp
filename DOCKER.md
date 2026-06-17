# Docker — запуск CLT пайплайна на нескольких GPU

Каноничный пайплайн — `base_clt_recon_fidelity_v2` (2048 фич, 3 стадии).
Степень параллелизма задаётся переменной `GPUS` (число процессов torchrun),
отдельных конфигов под каждое число GPU нет.

## Требования

- Docker >= 24
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- 1+ GPU. Для multi-GPU DDP — несколько GPU на одном узле (NVLink/PCIe).

---

## Быстрый старт

### 1. Собрать образ

```bash
docker compose build
```

Образ на базе `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel` (PyTorch 2.4.1,
CUDA 12.1, NCCL внутри). Прикладные зависимости ставятся из `requirements.txt`
(torch из базового образа не переустанавливается).

### 2. Полный пайплайн на N GPU

Тренировка (3 стадии, DDP на N GPU) → eval с контрольными бейзлайнами →
Deep Trace:

```bash
GPUS=4 docker compose run --rm pipeline
```

`GPUS` задаёт и число резервируемых карт, и `--nproc_per_node` для torchrun
(сервисы `train`/`pipeline` берут ровно `GPUS` GPU; `eval`/`deep-trace` — 1).
Какие именно карты использовать — `CUDA_VISIBLE_DEVICES`.

### 3. Только тренировка (одна стадия)

```bash
# stage 1 на 2 GPU
GPUS=2 docker compose run --rm train

# другая стадия / конфиг
GPUS=2 CONFIG=configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v1.yaml \
  docker compose run --rm train
```

### 4. Eval и Deep Trace по готовому чекпоинту

```bash
docker compose run --rm eval          # replacement metrics + baselines
docker compose run --rm deep-trace    # circuit suite
```

---

## Сервисы

| Сервис       | Что делает                                              | GPU      |
|--------------|---------------------------------------------------------|----------|
| `pipeline`   | train (3 стадии) → eval+baselines → Deep Trace          | N (train)|
| `train`      | одна стадия тренировки, конфиг через `CONFIG`           | N        |
| `eval`       | replacement metrics + zero/mean/random-CLT baselines    | 1        |
| `deep-trace` | Deep Trace stage-2 prompt suite                         | 1        |

`pipeline` принимает флаги скрипта 11, например продолжить без тренировки:

```bash
GPUS=4 docker compose run --rm pipeline \
  bash scripts/11_run_final_training_pipeline.sh --gpus 4 --skip-train
```

---

## Кастомный запуск / resume

```bash
# продолжить с конкретного чекпоинта
GPUS=2 docker compose run --rm train \
  bash scripts/run.sh --gpus 2 \
  --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v1.yaml \
  --continue-from outputs/base_clt_recon_fidelity_v2/clt_step_5000.pt
```

`--continue-from` переопределяет `training.init_from_checkpoint` через env
`CLT_OVERRIDE_INIT_CHECKPOINT`, который читает `train_clt.py`.

---

## Volume-маунты

| Путь в контейнере               | Источник на хосте        | Назначение                       |
|---------------------------------|--------------------------|----------------------------------|
| `/workspace/outputs`            | `./outputs`              | чекпоинты, метрики, Deep Trace   |
| `/workspace/configs`            | `./configs`              | конфиги (live-редактирование)    |
| `/workspace/scripts`            | `./scripts`              | скрипты (live-редактирование)    |
| `/workspace/.cache/huggingface` | Docker volume `hf_cache` | веса модели, датасеты, токен-кэш |

`outputs/` на хосте — артефакты доступны после остановки контейнера.
HF-кэш в Docker volume — модель и датасет не качаются заново при пересборке.

---

## Параметры стадий (per-GPU)

Батч в конфиге — **на один GPU**; эффективный батч = `batch_size_sequences × N`.

| Стадия      | Конфиг                                            | batch/GPU | lr      |
|-------------|---------------------------------------------------|-----------|---------|
| stage 1     | `..._recon_fidelity_v2.yaml`                      | 16        | 2e-4    |
| stage 2     | `..._recon_fidelity_v2_continue_v1.yaml`          | 16        | —       |
| stage 3     | `..._recon_fidelity_v2_continue_v2.yaml`          | 16        | 5e-5    |

При росте N эффективный батч растёт — при необходимости масштабируй lr под свой N.

---

## Мониторинг

```bash
watch -n 2 nvidia-smi

GPUS=4 docker compose run -d --name clt-pipe pipeline
docker logs -f clt-pipe
```

---

## Типичные ошибки

**`NCCL error: unhandled cuda error`** — несовместимость системного NCCL.
`run.sh` ставит `LD_LIBRARY_PATH` на bundled-NCCL из PyTorch; если не помогло:
```bash
DIST_BACKEND=gloo GPUS=2 docker compose run --rm train
```

**`Expected all tensors to be on the same device`** — `LOCAL_RANK` не передан:
запускай multi-GPU через `run.sh`/torchrun, не напрямую `python3`.

**Нет GPU в контейнере** — проверь `nvidia-container-toolkit` и
`docker info | grep -i runtime` (должен быть `nvidia`).
