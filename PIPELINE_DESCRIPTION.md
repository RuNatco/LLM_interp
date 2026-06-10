# Описание пайплайна обучения CLT

Проект обучает **Cross-Layer Transcoder (CLT)** — разреженный автоэнкодер поверх MLP-слоёв
модели Qwen2.5-0.5B. Цель: заменить все 24 MLP-блока реконструкциями CLT без значимой
потери качества предсказаний, а затем использовать полученное пространство фич для
механистической интерпретируемости через Deep Trace графы.



## Запуск поколения 2 (4096 фич)

```bash
# Stage 1 — обучение с нуля
python3 scripts/01_train_clt.py --config configs/qwen2_5_0_5b_4096f_v1.yaml

# Stage 2 — continuation с пониженным lr
python3 scripts/01_train_clt.py --config configs/qwen2_5_0_5b_4096f_v1_continue.yaml

# Eval
python3 scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_4096f_v1_continue.yaml

# Deep Trace
python3 scripts/09_build_deep_trace_prompt_suite.py \
  --checkpoint outputs/4096f_v1_continue/clt_final.pt \
  --output-dir outputs/4096f_v1_continue/deep_trace_suite \
  --summary-output outputs/4096f_v1_continue/deep_trace_suite_summary.json \
  --max-feature-nodes 64 --top-error-nodes 8 --causal-top-k 8
```




## Шаг 1. Обучение Stage 1

**Скрипт:** `scripts/01_train_clt.py`
**Config:** `configs/qwen2_5_0_5b_4096f_v1.yaml`

| Параметр | Значение | Зачем |
|---|---|---|
| `features_per_layer` | 4096 | Вдвое больше фич чем в baseline — больше ёмкость для разреженного представления |
| `max_train_tokens` | 200M | В 4× больше данных — нужно для сходимости большей модели |
| `max_optimizer_steps` | 30000 | Основной длинный прогон |
| `lr` | 0.0002 | Стандартный lr для обучения с нуля |
| `lr_scheduler` | cosine, warmup 500 шагов, `min_lr_ratio=0.1` | Warmup защищает от взрыва градиентов в начале; cosine decay плавно снижает lr к концу |
| `sparsity_warmup_steps` | 1000 | Первые 1000 шагов λ нарастает с 0 до 5e-5 — даёт модели сначала выучить реконструкцию, потом добавляет штраф |
| `lambda_sparsity` | 0.00005 | Целевой вес штрафа за ненулевые фичи |
| `weight_decay` | 0.01 | L2-регуляризация декодеров, предотвращает рост норм |
| `gradient_accumulation_steps` | 2 | Эффективный batch = 32 последовательности при batch_size=16 |
| `save_every` | 5000 | Редкое сохранение из-за большого размера чекпоинта (~6–8 GB) |
| `keep_last_checkpoints` | 1 | Хранить только последний промежуточный чекпоинт, чтобы не переполнить диск |
| `init_from_checkpoint` | нет | Обучение с нуля |

### Архитектура CLT, создаваемой на этом шаге

`CrossLayerTranscoder` (`src/qwen_clt/models/cross_layer_transcoder.py`):

- **Энкодеры:** 24 матрицы формы `[896, 4096]` (по одной на слой), инициализация Kaiming
- **Пороги JumpReLU:** тензор `[24, 4096]`, начальное значение `init_threshold=0.1`; обучаемые, задают минимальный уровень активации фичи
- **Декодеры:** треугольная матрица пар `src→tgt` для всех `tgt >= src` — итого 300 матриц формы `[4096, 896]`, инициализация Normal(0, 0.02); каждый src-слой может влиять на все последующие tgt-слои
- **Bias декодеров:** 24 вектора формы `[896]`, по одному на каждый tgt-слой

Суммарно параметров (без Adam state): ~1.2 млрд (декодеры 300 × 4096 × 896 доминируют).

### Что делается внутри

`scripts/01_train_clt.py` → `train_clt(cfg)` в `src/qwen_clt/training/train_clt.py`:

**Подготовка:**

1. **`load_qwen_model_and_tokenizer`** (`models/qwen_hooks.py`) — загружает Qwen2.5-0.5B
   в bfloat16 на GPU; все параметры Qwen замораживаются (`requires_grad=False`),
   отключается kv-cache
2. **`CrossLayerTranscoder`** — создаётся с нуля согласно архитектуре выше
3. **`AdamW`** — оптимизатор только для параметров CLT
4. **`build_lr_scheduler`** — `LambdaLR`: линейный warmup 500 шагов → cosine decay до `lr × 0.1`

**Тренировочный цикл (каждый optimizer step):**

5. **`iter_token_batches`** (`data/text_dataset.py`) — потоковая токенизация WikiText-103:
   текст нарезается окнами по 128 токенов, неполные хвосты отбрасываются;
   при `gradient_accumulation_steps=2` аккумулируются 2 микро-батча перед шагом

6. **`QwenMLPHookCollector`** (`models/qwen_hooks.py`) — за один `@torch.no_grad()` forward pass
   через Qwen собирает по всем 24 слоям:
   - `mlp_inputs[l]` — вход в `layer.mlp`, то есть `post_attention_layernorm(hidden_state)`, форма `[batch, seq, 896]`
   - `mlp_outputs[l]` — выход `layer.mlp`, форма `[batch, seq, 896]`

7. **`clt(mlp_inputs)`** — прямой проход CLT:
   - Для каждого слоя `src`: `pre[src] = mlp_inputs[src] @ encoder[src] + bias_enc[src]`
   - `features[src] = JumpReLU(pre[src], threshold[src])` — обнуляет значения ниже порога
   - Для каждого `tgt`: `recon[tgt] = bias_dec[tgt] + Σ_{src≤tgt} features[src] @ decoder[src→tgt]`

8. **Функции потерь** (`training/losses.py`):
   - `reconstruction_loss` = среднее по слоям `normalized_MSE(recon[tgt], mlp_outputs[tgt])`,
     где `normalized_MSE = MSE / Var(mlp_outputs[tgt])`
   - `tanh_sparsity_loss` = среднее по слоям и фичам `tanh(c · |feature| · decoder_norm)`;
     `decoder_norm[src][f]` = суммарная L2-норма строки `f` декодера `src` по всем tgt;
     в первые 1000 шагов умножается на `step / 1000`
   - `loss = reconstruction_loss + λ × tanh_sparsity_loss`

9. `loss.backward()` → gradient clip по норме 1.0 → `optimizer.step()` → `scheduler.step()`

**Периодические события:**

10. Каждые 50 шагов — логирование в `metrics.jsonl`:
    `l0_mean` (среднее число активных фич на токен), `nmse_mean` (средний normalized MSE),
    текущий lr, значения по каждому слою отдельно
11. Каждые 500 шагов — eval на test split WikiText-103 (400k токенов):
    те же `l0` и `nmse` без обновления весов
12. Каждые 5000 шагов — сохранение чекпоинта через `save_checkpoint` (`utils/io.py`):
    атомарная запись (сначала `.tmp`, затем `replace`), старые чекпоинты удаляются
    кроме последнего 1
13. После последнего шага — сохранение `clt_final.pt`

**Формат чекпоинта:**
```python
{
    "cfg": { ... },              # полный yaml-конфиг
    "model_state_dict": { ... }, # веса CLT
    "optimizer_state_dict": { ... },
    "step": int,
}
```

### Что получается

```
outputs/4096f_v1/
  clt_final.pt          # финальный чекпоинт после 30000 шагов
  clt_step_25000.pt     # единственный хранимый промежуточный (keep_last=1)
  metrics.jsonl         # лог train/eval метрик по шагам
```

---

## Шаг 2. Обучение Stage 2 — continuation


**Config:** `configs/qwen2_5_0_5b_4096f_v1_continue.yaml`

| Параметр | Значение |
|---|---|
| `init_from_checkpoint` | `outputs/4096f_v1/clt_final.pt` |
| `max_optimizer_steps` | 5000 |
| `lr` | 0.00002 (в 10× меньше, чем Stage 1) |
| `lr_scheduler` | cosine, warmup 100 шагов |
| `sparsity_warmup_steps` | 0 (полный штраф сразу — модель уже сошлась) |
| `save_every` | 1000 |
| `keep_last_checkpoints` | 3 |

### Что делается

Внутри `train_clt.py` вызывается `load_initial_clt_weights(clt, checkpoint_path)` —
загружает `model_state_dict`, **не** загружает состояние оптимизатора. Обучение
начинается заново с новыми гиперпараметрами (lr, scheduler, seed).

### Что получается

```
outputs/4096f_v1_continue/
  clt_final.pt          ← ФИНАЛЬНЫЙ ЧЕКПОИНТ 
  clt_step_*.pt
  metrics.jsonl
```

---

## Шаг 4. Оценка Replacement Model

**Скрипт:** `scripts/02_eval_replacement_model.py`

```bash
python3 scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_4096f_v1_continue.yaml
```

Дополнительные флаги для ручного запуска:

| Флаг | Эффект |
|---|---|
| `--checkpoint PATH` | Переопределить путь к чекпоинту CLT |
| `--output PATH` | Переопределить путь к выходному JSON |
| `--max-eval-tokens N` | Переопределить число токенов на eval |
| `--batch-size-sequences N` | Переопределить размер батча |

### Что загружается

- **Qwen2.5-0.5B** — загружается заново в bfloat16, замораживается
- **CLT** из `outputs/4096f_v1_continue/clt_final.pt` через `load_autoencoder_from_checkpoint`
  (`replacement/loader.py`): читает `cfg` и `model_state_dict` из чекпоинта,
  восстанавливает архитектуру (n_layers, d_model, features_per_layer),
  загружает веса со `strict=False` с выводом предупреждений при несовпадении ключей
- **WikiText-103**, 50k токенов из train split (batch_size=64 последовательностей по 128 токенов)
- Токенизатор для разрешения `target_logit_diff`: определяет id токенов `" increase"` и `" decrease"`

### Что делается внутри

**Прогон с заменой:**

1. **`LayerReplacementHook`** (`replacement/hooks.py`) навешивается на все 24 `layer.mlp`
   через `register_forward_hook`. Внутри хука для каждого слоя `tgt`:
   - Кешируется `mlp_input[tgt]` (post-attention layernorm output)
   - CLT энкодирует все src ≤ tgt: `features[src] = JumpReLU(mlp_input[src] @ encoder[src])`
   - Реконструкция: `recon[tgt] = bias[tgt] + Σ_{src≤tgt} features[src] @ decoder[src→tgt]`
   - Оригинальный output `layer.mlp` заменяется на `recon[tgt]`
2. Forward pass Qwen с хуками → `replacement_logits` формы `[batch, seq, vocab]`

**Прогон оригинала:**

3. Forward pass Qwen без хуков → `original_logits` формы `[batch, seq, vocab]`

**Подсчёт метрик** — `compute_replacement_metrics` (`replacement/metrics.py`):

4. Для каждого батча по всем активным токенам (по attention mask):
   - **`kl_div`** = `Σ p_orig(t) × log(p_orig(t) / p_repl(t))` усреднённый по токенам —
     измеряет насколько распределение вероятностей replacement отличается от оригинала
   - **`top1_agreement`** = доля токенов, где `argmax(original) == argmax(replacement)` —
     самая интерпретируемая метрика: совпадает ли следующий предсказанный токен
   - **`mean_abs_logit_diff`** = средний `|original_logit - replacement_logit|` по всему словарю
   - **`last_token_*`** — те же три метрики, но только по последнему токену каждой
     последовательности; важнее, так как именно на нём модель делает реальное предсказание
   - **`target_logit_diff_mae`** = MAE между `logit(" increase") - logit(" decrease")` у
     оригинала и у replacement на целевой позиции (`target_pos=-1`); проверяет, сохраняет ли
     replacement семантическое направление предсказания

5. Метрики аккумулируются по батчам с весом `num_tokens`, финальное значение — взвешенное среднее

### Пороги gate

| Метрика | Порог | Что означает провал |
|---|---|---|
| `last_token_top1_agreement` | ≥ 0.50 | Replacement угадывает следующий токен реже чем в половине случаев |
| `kl_div` | ≤ 1.40 | Распределение вероятностей сильно искажено по всем позициям |
| `last_token_kl_div` | ≤ 1.35 | То же, но на последнем токене — критичнее для генерации |
| `target_logit_diff_mae` | ≤ 0.55 | CLT не сохраняет семантику " increase" vs " decrease" |

### Что получается

```
outputs/4096f_v1_continue/replacement_eval_metrics.json
```

Структура файла:
```json
{
  "metrics": {
    "num_tokens": 50000,
    "num_sequences": 390,
    "kl_div": ...,
    "logit_mse": ...,
    "top1_agreement": ...,
    "mean_abs_logit_diff": ...,
    "last_token_kl_div": ...,
    "last_token_logit_mse": ...,
    "last_token_top1_agreement": ...,
    "last_token_mean_abs_logit_diff": ...,
    "target_logit_diff_original_mean": ...,
    "target_logit_diff_replacement_mean": ...,
    "target_logit_diff_mae": ...,
    "target_logit_diff_mse": ...
  }
}
```

Просмотр результатов:
```bash
cat outputs/4096f_v1_continue/replacement_eval_metrics.json | python3 -m json.tool
```

---

## Шаг 5. Построение Deep Trace Suite

**Скрипт:** `scripts/09_build_deep_trace_prompt_suite.py`

```bash
python3 scripts/09_build_deep_trace_prompt_suite.py \
  --checkpoint outputs/4096f_v1_continue/clt_final.pt \
  --output-dir outputs/4096f_v1_continue/deep_trace_suite \
  --summary-output outputs/4096f_v1_continue/deep_trace_suite_summary.json \
  --max-feature-nodes 64 \
  --top-error-nodes 8 \
  --causal-top-k 8
```

Параметры командной строки:

| Параметр | Значение | Зачем |
|---|---|---|
| `--max-feature-nodes` | 64 | Максимум фич-кандидатов на граф — баланс полноты и скорости |
| `--top-error-nodes` | 8 | Топ-8 слоёв по норме ошибки реконструкции добавляются как MLPErrorNode |
| `--causal-top-k` | 8 | Для каждой фичи строятся рёбра к топ-8 residual-позициям |
| `--min-activation` | 0.0 | Минимальный порог активации для включения фичи в граф |
| `--positive` | `" increase"` | Целевой токен «правильного» направления |
| `--negative` | `" decrease"` | Целевой токен «неправильного» направления |
| `--strict-fidelity` | не задан | Прерывать при провале fidelity gate (по умолчанию — предупреждение) |

### Что загружается

- **`ProxyQwenReplacementModel.from_checkpoint`** (`models/proxy_replacement_model.py`) —
  загружает Qwen2.5-0.5B и CLT вместе; CLT переводится в `.eval()` с `strict=True`
  при загрузке state dict (любое несовпадение ключей — ошибка)
- **Промпты** — по умолчанию 5 экономических примеров:
  ```
  "Demand is greater than generation, so the price will"
  "Demand is lower than supply, so the price will"
  "Fuel supply falls while demand stays high, so the price will"
  "Generation exceeds demand, so the price will"
  "A product becomes scarce while buyers still need it, so the price will"
  ```
  Можно передать свои через `--prompt TEXT` или `--prompts-file prompts.json`
- **`replacement_eval_metrics.json`** из директории чекпоинта — для fidelity gate

### Что делается внутри

`build_deep_trace_graph` (`deep_trace/build.py`) выполняется для каждого промпта:

**1. Fidelity gate** (`attribution/fidelity.py`)

Читает `replacement_eval_metrics.json`, проверяет `last_token_top1_agreement ≥ 0.3`
и `target_logit_diff_mae ≤ 2.0`. При провале — предупреждение в логе;
при `--strict-fidelity` — исключение `RuntimeError`.

**2. Сбор активаций — `QwenDeepTraceCollector`** (`deep_trace/cache.py`)

Расширенный forward pass с хуками на каждый слой, собирает:
- `residual_inputs[l]` — вход слоя `l` до attention (residual stream)
- `attention_outputs[l]` — выход self-attention
- `attention_head_outputs[l]` — выходы каждой головы отдельно (форма `[1, seq, n_heads, head_dim]`)
- `mlp_inputs[l]`, `mlp_outputs[l]` — вход/выход MLP
- `logits` — финальные логиты

Одновременно запускается CLT на собранных `mlp_inputs` → получаем `features[l]` и `recon[l]`.

**3. Целевой вектор** (`attribution/targets.py`)

```python
target_vector = lm_head.weight[token_id(" increase")] - lm_head.weight[token_id(" decrease")]
```

Это направление в пространстве d_model, проекция на которое соответствует разнице
логитов между двумя токенами.

**4. Отбор фич-кандидатов — `select_feature_candidates`**

Для каждой фичи `f` в слое `src` на целевой позиции `pos`:
- `direct_score = features[src][pos, f] × (decoder[src→src][f] · target_vector)`
- Берутся топ-64 фичи по `|direct_score|`

**5. Каузальные аблации — `feature_intervention`** (`models/proxy_replacement_model.py`)

Для каждой фичи-кандидата выполняется **2 forward pass** через Qwen с `LayerReplacementHook`:

- **Baseline pass:** CLT replacement без вмешательства → `replacement_logits`
- **Intervened pass:** CLT replacement с `FeatureIntervention(layer, pos, feature_idx, value=0.0)` — обнуляет одну фичу → `intervened_logits`

`causal_effect = logit_diff(replacement) - logit_diff(intervened)` — насколько эта фича
в реальности двигает logit-разность.

Строятся рёбра в граф:
- `causal_ablation_effect` — ребро `feature → target` с весом `causal_effect`
- `causal_feature_to_residual_delta` — рёбра `feature → residual[layer]` с весом
  `(intervened_residual[l] - baseline_residual[l]) · target_vector` для топ-8 слоёв

**6. Проверка знаков — `ablation_matches_proxy_sign`** (`attribution/validation.py`)

```python
sign_match = (proxy_score > 0 and causal_effect < 0) or (proxy_score < 0 and causal_effect > 0)
```

Логика: если фича *помогает* предсказать " increase" (proxy > 0), то её обнуление
*снижает* logit-разность (causal_effect < 0). Несовпадение знаков означает,
что proxy-оценка не отражает реальный каузальный вклад.

**7. MLPErrorNode** — для каждого слоя `l` считается `error_norm = |mlp_output[l] - recon[l]|²`.
Топ-8 слоёв по `|error_vector · target_vector|` добавляются в граф как узлы ошибки
с ребром `replacement_error_projection`.

**8. Сериализация** (`deep_trace/graph.py`) — граф пишется в JSON:
```json
{
  "nodes": [ { "id": "feature:L3:P7:F42", "type": "CLTFeatureNode", "value": 1.23, ... } ],
  "edges": [ { "source": "feature:L3:P7:F42", "target": "target:0", "score": -0.87,
               "kind": "causal_ablation_effect" } ],
  "metadata": { "prompt": "...", "checkpoint": "...", "trace_kind": "deep_trace_stage2" }
}
```

**9. Агрегация по промптам** — `summarize_deep_trace_payload` (`deep_trace/summary.py`):
подсчитывает `mean_sign_match`, `median_sign_match`, `mean_mlp_error_norm` по всем промптам,
топ-рёбра и топ-узлы ошибки.

### Пороги gate

| Метрика | Порог | Что означает провал |
|---|---|---|
| `replacement fidelity gate` | passed | CLT слишком плохо воспроизводит оригинал — графы ненадёжны |
| `mean sign_match` | ≥ 0.90 | Proxy-оценки не предсказывают реальные каузальные эффекты |
| `median sign_match` | = 1.00 | Большинство фич не проходят знаковый тест |
| `mean MLP error norm` | ≤ 7.00 | CLT слишком плохо реконструирует MLP-выходы в среднем |

### Что получается

```
outputs/4096f_v1_continue/
  deep_trace_suite/
    prompt_0.json         # граф для каждого промпта
    prompt_1.json
    prompt_2.json
    prompt_3.json
    prompt_4.json
  deep_trace_suite_summary.json   # агрегат по всем промптам
```

Просмотр сводки:
```bash
cat outputs/4096f_v1_continue/deep_trace_suite_summary.json | python3 -m json.tool
```

---



## Финальные артефакты


```
outputs/4096f_v1_continue/
  clt_final.pt
  replacement_eval_metrics.json
  deep_trace_suite_summary.json
  deep_trace_suite/prompt_0.json ... prompt_4.json
```

---


## Схема потока данных

```
WikiText-103 (200M токенов)
        │
        ▼
[Qwen2.5-0.5B] ──forward pass──► MLP inputs[0..23]
        │                         MLP outputs[0..23]
        │
        ▼
[CrossLayerTranscoder]
  encode: mlp_input[src] ──JumpReLU──► features[src]   (4096 фич на слой)
  decode: Σ features[src] @ decoder[src→tgt] ──► recon[tgt]
        │
        ├─► reconstruction loss (normalized MSE)
        └─► sparsity loss (tanh, warmup 1000 шагов)
                 │
                 ▼
            AdamW + cosine LR
                 │
           Stage 1: 30000 шагов
                 │
           Stage 2: 5000 шагов (lr × 0.1)
                 │
                 ▼
          clt_final.pt (outputs/4096f_v1_continue/)
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
[Replacement eval]    [Deep Trace]
LayerReplacementHook  ProxyQwenReplacementModel
заменяет MLP outputs  каузальные аблации фич
в реальном forward    feature_intervention × N
       │                   │
       ▼                   ▼
replacement_eval_      deep_trace_suite/
metrics.json               │
       │                   │
       └─────────┬─────────┘
                 ▼
            gate checks
  last_token_top1_agreement ≥ 0.50
  kl_div ≤ 1.40
  sign_match ≥ 0.90
```
