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
  qwen2_5_0_5b_base_clt_fidelity_v1.yaml
  qwen2_5_0_5b_base_clt_fidelity_v2.yaml
  qwen2_5_0_5b_base_clt_fidelity_v3.yaml
  qwen2_5_0_5b_base_clt_fidelity_v4.yaml
  qwen2_5_0_5b_base_clt_late_loss_v1.yaml
  qwen2_5_0_5b_base_clt_late_target_v1.yaml
  qwen2_5_0_5b_base_clt_late_target_v2.yaml
  qwen2_5_0_5b_base_clt_late_target_v3.yaml
  qwen2_5_0_5b_base_clt_recon_fidelity_v1.yaml
  qwen2_5_0_5b_base_clt_recon_fidelity_v2.yaml
  qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v1.yaml
  qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v2.yaml
  qwen2_5_0_5b_instruct_clt_v0.yaml
  qwen2_5_0_5b_instruct_clt_fidelity_v2.yaml

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
  deep_trace/
    cache.py
    graph.py
    build.py
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
  07_build_deep_trace_graph.py
  08_summarize_deep_trace_graph.py
  09_build_deep_trace_prompt_suite.py
```

## Replacement abstractions

В проекте разведены две разные роли replacement.

`src/qwen_clt/replacement/` — hook-based replacement path. Он используется для
оценки replacement logits внутри настоящего forward pass: KL, logit MSE,
top-1 agreement и другие метрики из `scripts/02_eval_replacement_model.py`.
Метрики считаются по attention mask, поэтому padding не загрязняет результат.
JSON дополнительно сохраняет last-token метрики и, если задано в YAML,
target logit-difference fidelity.
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

`scripts/03_build_attribution_graph.py` и
`scripts/06_validate_attribution_graph.py` читают sibling-файл
`replacement_eval_metrics.json` рядом с checkpoint и печатают fidelity warnings.
С флагом `--strict-fidelity` низкая replacement fidelity останавливает graph или
validation run. Без этого флага graph всё ещё можно строить как diagnostic
proxy, но его нельзя описывать как faithful circuit.

## Deep Trace stage 1

`scripts/07_build_deep_trace_graph.py` — первый шаг от proxy attribution к
более глубокому replacement-model tracing. Основной baseline для этого этапа —
`qwen2_5_0_5b_base_clt_fidelity_v2`, потому что он лучше всего сохраняет
target direction и общий logit landscape среди текущих запусков.
`scripts/08_summarize_deep_trace_graph.py` печатает top error nodes и causal
edges из готового graph JSON. `scripts/09_build_deep_trace_prompt_suite.py`
строит Deep Trace graphs для набора prompts, чтобы результат не зависел от
одного примера.

Здесь `stage 1` означает версию tracing-архитектуры, а не CLT checkpoint
`base_clt_fidelity_v1`. Основной checkpoint для Deep Trace stage 1 — v2.

```text
proxy attribution
-> validated proxy graph
-> cached replacement trace
-> typed circuit graph with error nodes
-> causal pruning
-> frontend-compatible graph
```

`Deep Trace stage 1` уже добавляет:

- activation cache для residual stream, attention outputs, MLP outputs,
  layernorm inputs/outputs, replacement errors и logits;
- typed graph nodes: `CLTFeatureNode`, `AttentionHeadNode`, `MLPErrorNode`,
  `ResidualStreamNode`, `LayerNormNode`, `LogitTargetNode`;
- явные error nodes для `original_mlp_output - clt_reconstruction`;
- replacement-conditioned CLT features, то есть трассируется именно
  replacement model, а не только original Qwen activations;
- decoder-write edges от features к residual nodes;
- causal ablation edges для top-k feature nodes;
- graph JSON с node types, edge kinds, scores, fidelity metadata и validation
  metadata.

Ограничение остаётся: это ещё не full path attribution. Attention в v1 хранится
как layer-level output, не как per-head decomposition. Deep graph нужно читать
вместе с replacement fidelity и causal validation, иначе он может объяснять
ошибки replacement model, а не поведение Qwen.

## Causal validation

`scripts/06_validate_attribution_graph.py` проверяет top-k узлов proxy graph
через in-forward feature ablation. Для каждого узла скрипт сохраняет значение
feature до/после intervention, proxy causal estimate и фактический causal
effect:

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
В summary отчёта также есть `proxy_causal_pearson`,
`ablation_sign_match_rate` и средняя абсолютная proxy/causal ошибка.

Для ручной проверки конкретной feature лучше выбирать узел из validation report,
а не копировать старые индексы feature вручную:

```bash
python scripts/04_run_feature_interventions.py \
  --checkpoint outputs/base_clt_v0/clt_final.pt \
  --prompt "Demand is greater than generation, so the price will" \
  --validation-report outputs/base_clt_v0/price_graph_validated.json \
  --rank 1 \
  --select-by abs_causal_effect \
  --value 0.0 \
  --positive " increase" \
  --negative " decrease"
```

## Training budget

Основные YAML используют `training.max_optimizer_steps`, а не старый
`max_steps`. При `gradient_accumulation_steps: 8` это означает, что один
optimizer step соответствует восьми micro-batches. Старый `max_steps` всё ещё
поддерживается как legacy micro-batch limit, но новые эксперименты лучше
задавать через `max_optimizer_steps`.

Текущие baseline-конфиги выровнены для честного сравнения base/instruct:

```text
features_per_layer: 256
lambda_sparsity: 0.00005
max_optimizer_steps: 1500
```

Если replacement fidelity всё ещё низкая, следующий дорогой прогон стоит
делать с `features_per_layer: 512` и `max_optimizer_steps: 3000`.
Для этого добавлен отдельный config:

```text
configs/qwen2_5_0_5b_base_clt_fidelity_v1.yaml
```

Он пишет результаты в `outputs/base_clt_fidelity_v1`, чтобы не смешивать
high-fidelity attempt с основным `base_clt_v0`.

После успешного `v1` можно пробовать более дорогой `v2`:

```text
configs/qwen2_5_0_5b_base_clt_fidelity_v2.yaml
```

Он использует `features_per_layer: 1024`, `max_optimizer_steps: 10000` и
`lambda_sparsity: 0.00002`; цель — приблизиться к
`last_token_top1_agreement` в диапазоне `0.4-0.6`.

Для Instruct-линии есть зеркальный high-fidelity config:

```text
configs/qwen2_5_0_5b_instruct_clt_fidelity_v2.yaml
```

Он использует ту же CLT capacity, но модель
`Qwen/Qwen2.5-0.5B-Instruct` и `chat_template: true`. Deep Trace scripts
автоматически применяют instruct chat template из checkpoint config.

Если `v2` улучшает KL и target logit-difference, но last-token top-1 остаётся
около `0.3`, следующий шаг — не просто увеличивать capacity, а менять цель
обучения:

```text
configs/qwen2_5_0_5b_base_clt_fidelity_v3.yaml
```

`v3` сохраняет capacity `v2`, но включает layerwise activation/target
normalization внутри CLT. Replacement hook и proxy attribution используют те же
stats из checkpoint, поэтому реконструкция по-прежнему подставляется в raw
`mlp_output` scale.

```text
configs/qwen2_5_0_5b_base_clt_fidelity_v4.yaml
```

`v4` добавляет лёгкую last-token logit distillation: на каждом micro-batch
replacement forward сравнивается с original Qwen только по teacher top-k logits
на последней активной позиции. Это дороже, чем `v3`, но должно бить прямо в
`last_token_top1_agreement`.

Deep Trace suite на base v2 показал, что replacement errors системно
концентрируются в поздних слоях `L18-L23`, особенно `L20-L21`. Для проверки
этой гипотезы добавлен targeted run:

```text
configs/qwen2_5_0_5b_base_clt_late_loss_v1.yaml
```

Он сохраняет capacity `v2`, но взвешивает reconstruction loss по слоям:
`L18/L19/L22/L23` получают вес `2.0`, `L20/L21` — вес `3.0`. Цель — снизить
late-layer `MLPErrorNode` projections и улучшить target-direction fidelity без
агрессивной distillation.

Так как `late_loss_v1` улучшил causal sign agreement, но не улучшил
`target_logit_diff_mae` и выявил новый bottleneck в `L23`, добавлены
target-aware runs:

```text
configs/qwen2_5_0_5b_base_clt_late_target_v1.yaml
configs/qwen2_5_0_5b_base_clt_late_target_v2.yaml
configs/qwen2_5_0_5b_base_clt_late_target_v3.yaml
```

Они сохраняют late-layer weighting, усиливают `L23` до веса `4.0` и добавляют
target-aware loss на сохранение:

```text
logit(" increase") - logit(" decrease")
```

`late_target_v2` с `weight: 0.02` показал negative result: target MAE слегка
улучшился, но global replacement fidelity разрушилась. `late_target_v3` с
`weight: 0.001`, `every_n_micro_steps: 10` тоже ухудшил fidelity и target MAE.
Поэтому target direction теперь используется как eval/validation metric, а не
как training loss.

Общие fidelity runs:

```text
configs/qwen2_5_0_5b_base_clt_recon_fidelity_v1.yaml
configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2.yaml
```

Они не используют target-loss, logit distillation или normalization. Вместо
этого они увеличивают CLT capacity, снижают sparsity weight до `0.00001`,
увеличивают training budget до `15000` optimizer steps и проверяют, можно ли
улучшить replacement fidelity более общей реконструкцией. `recon_fidelity_v2`
с `features_per_layer: 2048` стал сильным baseline для continuation.
`recon_fidelity_v2_continue_v1` дообучает этот checkpoint ещё 5000 шагов через
`training.init_from_checkpoint`, сохраняя результат в отдельный output dir.
`recon_fidelity_v2_continue_v2` продолжает уже от `continue_v1` ещё 3000 шагов
с `lr: 0.00005` и является текущим лучшим replacement baseline. Его Deep Trace
prompt suite прошёл fidelity gate и дал mean sign match около `0.943`.

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
4. После обучения оценивается replacement fidelity: особенно
   `last_token_top1_agreement`, `last_token_kl_div` и target logit-difference.
5. Layerwise/prefix diagnostics показывают, какие слои ломают full replacement.
6. Для нескольких prompt-задач строятся attribution graphs.
7. Top features проверяются через ablation/steering.

Полный набор команд для последовательного запуска описан в
`EXPERIMENT_PIPELINE.md`.

## Важное ограничение

Это research scaffold. Он задаёт архитектуру проекта и минимальные реализации. Для тяжёлого обучения нужно будет добавить:

- распределённое обучение / gradient checkpointing;
- chunked triangular decoder;
- sparse матричные операции;
- более строгую совместимость с frontend `circuit-tracer`;
- сохранение больших activation caches в memmap/safetensors.
