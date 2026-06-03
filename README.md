# Qwen CLT Circuit Baseline

Проект-скелет для обучения **Cross-Layer Transcoder (CLT)** под две модели:

- `Qwen/Qwen2.5-0.5B` — технический baseline на base-модели;
- `Qwen/Qwen2.5-0.5B-Instruct` — прикладной baseline, ближе к будущей domain-tuned instruct-модели.

Цель проекта: воспроизвести ключевую идею circuit tracing baseline:  
**Qwen → MLP inputs/outputs → CLT → replacement model → attribution graph → feature interventions**.

## Структура

```text
configs/
  qwen2_5_0_5b_base_clt_v0.yaml
  qwen2_5_0_5b_instruct_clt_v0.yaml

src/qwen_clt/
  data/
    text_dataset.py
  models/
    qwen_hooks.py
    cross_layer_transcoder.py
    proxy_replacement_model.py
  replacement/
    hooks.py
    loader.py
    metrics.py
  training/
    losses.py
    metrics.py
    train_clt.py
  attribution/
    targets.py
    graph.py
    attribute.py
    prune.py
  interventions/
    feature_intervention.py
  utils/
    config.py
    seed.py
    io.py

scripts/
  01_train_clt.py
  02_eval_replacement_model.py
  03_build_attribution_graph.py
  04_run_feature_interventions.py
  05_visualize_attribution_graph_svg.py
  06_validate_attribution_graph.py
```

## Replacement abstractions

В проекте разведены две разные роли replacement.

`src/qwen_clt/replacement/` — hook-based replacement path. Он используется для
оценки replacement logits внутри настоящего forward pass: KL, logit MSE,
top-1 agreement и другие метрики из `scripts/02_eval_replacement_model.py`.
Хук ставится на `layer.mlp`: он читает `mlp_normed_input` из `inputs[0]` и
заменяет `mlp_output` реконструкцией CLT. Это соответствует тому, как CLT
обучается в `QwenMLPHookCollector`.

`src/qwen_clt/models/proxy_replacement_model.py` — proxy wrapper для анализа
CLT feature space. Он собирает исходные Qwen activations, кодирует их в CLT
features и используется для v0 attribution graphs. Для feature interventions
он запускает настоящий forward pass с CLT replacement hook и возвращает
`replacement_logits`, `intervened_logits` и `intervention_logit_delta`.
Эффект фичи нужно сравнивать с replacement baseline, потому что сама CLT
реконструкция может сдвигать logits относительно оригинальной Qwen.

## Attribution graphs

Текущие attribution graphs — это proxy, а не полноценный circuit tracing.
`src/qwen_clt/attribution/attribute.py` оценивает прямой вклад активной CLT
feature в target logit direction через decoder rows:

```text
score = feature_activation * dot(sum_outgoing_decoder_rows, target_logit_direction)
```

Этот proxy не моделирует attention-mediated paths, residual stream dynamics,
layernorm/error nodes и ошибки replacement model. Поэтому графы нужно
интерпретировать как гипотезы о важных features, а не как доказанные circuits.
Важные узлы следует проверять через feature interventions и сравнивать
`intervened_logits` с `replacement_logits`. Новые graph JSON сохраняют это
ограничение в поле `metadata`.

## Causal validation

`scripts/06_validate_attribution_graph.py` проверяет top-k узлов proxy graph
через in-forward feature ablation. Для каждого узла скрипт считает:

```text
causal_effect =
  logit_diff(intervened_logits) - logit_diff(replacement_logits)
```

Пример:

```bash
python scripts/06_validate_attribution_graph.py \
  --checkpoint outputs/base_clt_v0/clt_final.pt \
  --graph outputs/base_clt_v0/price_graph.json \
  --prompt "Demand is greater than generation, so the price will" \
  --positive " increase" \
  --negative " decrease" \
  --top-k 24 \
  --output outputs/base_clt_v0/price_graph_validated.json
```

Если proxy score положительный, то при ablation ожидается отрицательный
`causal_effect`: feature поддерживала target direction, а зануление её ослабило.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt
pip install -e .
```

## Быстрый запуск base baseline

```bash
python scripts/01_train_clt.py --config configs/qwen2_5_0_5b_base_clt_v0.yaml
```

## Быстрый запуск instruct baseline

```bash
python scripts/01_train_clt.py --config configs/qwen2_5_0_5b_instruct_clt_v0.yaml
```

## Логика эксперимента

1. Сначала запускается `base`.
2. Проверяется, что CLT обучается: падает NMSE, L0 не схлопывается в 0 и не становится слишком большим.
3. Затем тот же pipeline запускается на `instruct`.
4. После обучения строится replacement model.
5. Для нескольких prompt-задач строятся attribution graphs.
6. Top features проверяются через ablation/steering.

Полный набор команд для последовательного запуска описан в
`EXPERIMENT_PIPELINE.md`.

## Важное ограничение

Это research scaffold. Он задаёт архитектуру проекта и минимальные реализации. Для тяжёлого обучения нужно будет добавить:

- распределённое обучение / gradient checkpointing;
- chunked triangular decoder;
- sparse матричные операции;
- более строгую совместимость с frontend `circuit-tracer`;
- сохранение больших activation caches в memmap/safetensors.
