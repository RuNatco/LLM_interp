# Docker — запуск CLT обучения

## Требования

- Docker >= 24  
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (`nvidia-container-toolkit`)  
- 2 GPU ≥ 24 GB VRAM (рекомендовано); работает и на 1 GPU

---

## Быстрый старт

### 1. Собрать образ

```bash
docker compose build
```

Образ основан на `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel` — PyTorch 2.4.1, CUDA 12.1, NCCL 2.21 уже внутри. Сборка занимает ~2–3 минуты.

### 2. Обучение на 1 GPU

```bash
docker compose run --rm train-1gpu
```

Запускает `scripts/run.sh --gpus 1 --config configs/qwen2_5_0_5b_4096f_v1.yaml`.

### 3. Обучение на 2 GPU (DDP)

```bash
docker compose run --rm train-2gpu
```

Запускает `torchrun --nproc_per_node=2` с конфигом `qwen2_5_0_5b_4096f_v1_2gpu.yaml`.  
Эффективный батч: **64 seq × 256 tokens = 16 384 tokens/step**.

---

## Кастомный запуск

Переопределить команду можно через аргументы `docker compose run`:

```bash
# другой конфиг
docker compose run --rm train-2gpu \
  bash scripts/run.sh --gpus 2 --config configs/qwen2_5_0_5b_4096f_v1_continue.yaml

# продолжить с чекпоинта
docker compose run --rm train-2gpu \
  bash scripts/run.sh --gpus 2 \
  --config configs/qwen2_5_0_5b_4096f_v1_2gpu.yaml \
  --continue-from outputs/4096f_v1_2gpu/clt_step_5000.pt

# запустить eval
docker compose run --rm eval \
  --config configs/qwen2_5_0_5b_4096f_v1_2gpu.yaml
```

---

## Структура volume-маунтов

| Путь в контейнере            | Источник на хосте              | Назначение                         |
|------------------------------|--------------------------------|------------------------------------|
| `/workspace/outputs`         | `./outputs`                    | чекпоинты, метрики (сохраняются)   |
| `/workspace/configs`         | `./configs`                    | конфиги (live-редактирование)      |
| `/workspace/scripts`         | `./scripts`                    | скрипты (live-редактирование)      |
| `/workspace/.cache/huggingface` | Docker volume `hf_cache`   | веса модели, датасеты HF           |
| `/workspace/.cache/tokenized`  | Docker volume `tok_cache`   | pre-tokenized чанки (ускоряет restart) |

`outputs/` монтируется с хоста — чекпоинты доступны после остановки контейнера.  
HF-кэш в Docker volume — не скачивается заново при пересборке образа.

---

## Конфиги по режимам

| Файл                                      | Режим       | batch/GPU | seq_len | eff. batch | lr      |
|-------------------------------------------|-------------|-----------|---------|------------|---------|
| `qwen2_5_0_5b_4096f_v1.yaml`              | 1-GPU       | 16        | 128     | 32 seq     | 0.0002  |
| `qwen2_5_0_5b_4096f_v1_2gpu.yaml`         | 2-GPU Docker| 32        | 256     | 64 seq     | 0.0003  |
| `qwen2_5_0_5b_4096f_v1_continue.yaml`     | 1-GPU, finetune | 16    | 128     | 32 seq     | 0.00002 |

---

## Мониторинг GPU во время обучения

В отдельном терминале:

```bash
# утилизация всех GPU
watch -n 2 nvidia-smi

# логи контейнера в реальном времени (если запущен через --detach)
docker compose logs -f train-2gpu
```

Запуск в фоне:

```bash
docker compose run -d --name clt-train train-2gpu
docker logs -f clt-train
```

---

## Типичные ошибки

**`NCCL error: unhandled cuda error`**  
Система NCCL несовместима с CUDA runtime. `scripts/run.sh` автоматически ставит `LD_LIBRARY_PATH` на PyTorch-bundled NCCL — если ошибка всё равно есть, добавь в команду:
```bash
DIST_BACKEND=gloo docker compose run --rm train-2gpu
```

**`RuntimeError: Expected all tensors to be on the same device`**  
`LOCAL_RANK` не передаётся — убедись, что запуск через `torchrun` (через `scripts/run.sh`), не напрямую через `python3`.

**Нет GPU в контейнере**  
Проверь, что `nvidia-container-toolkit` установлен и `docker info | grep -i runtime` показывает `nvidia`.
