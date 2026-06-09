# Experiment Results Log

Этот файл фиксирует фактически проведённые эксперименты по CLT replacement
model и Deep Trace для `Qwen/Qwen2.5-0.5B` и
`Qwen/Qwen2.5-0.5B-Instruct`. Он отделён от `EXPERIMENT_PIPELINE.md`: pipeline
описывает команды запуска, а здесь собраны цели, изменения, результаты и
выводы по каждому запуску.

## Общая цель экспериментов

Проект исследует, можно ли заменить MLP-блоки модели Qwen с помощью
Cross-Layer Transcoder (CLT), а затем использовать CLT features для
интерпретации поведения модели через attribution graph, causal interventions и
Deep Trace stage 1.

Основная проверяемая цепочка:

```text
Qwen activations
-> CLT reconstruction of MLP outputs
-> hook-based replacement forward pass
-> replacement logits
-> feature attribution / interventions
-> Deep Trace graph with residual, layernorm, attention and error nodes
```

Главная методологическая идея: интерпретировать features можно только настолько
надёжно, насколько replacement model сохраняет поведение исходной модели. Если
replacement logits сильно отличаются от original logits, граф объясняет уже
ошибки replacement model, а не исходную Qwen.

## Основные метрики

`top1_agreement` показывает совпадение top-1 токена по всем непаддинговым
позициям.

`last_token_top1_agreement` показывает совпадение top-1 токена на последней
смысловой позиции последовательности. Для causal prompts это самая важная
дискретная метрика, потому что именно последний token position используется в
target direction.

`kl_div` и `last_token_kl_div` измеряют расхождение распределений logits между
original и replacement model. Чем ниже, тем лучше. Для circuit tracing эта
метрика часто важнее top1, потому что она отражает форму всего logit
landscape.

`mean_abs_logit_diff` и `last_token_mean_abs_logit_diff` измеряют средний
абсолютный сдвиг logits.

`target_logit_diff_mae` измеряет ошибку по направлению:

```text
logit(" increase") - logit(" decrease")
```

Эта метрика особенно важна для supply/demand prompts, потому что Deep Trace
строится вокруг target direction "increase vs decrease".

`causal sign match` в Deep Trace показывает, насколько знак causal ablation
effect согласуется с ожидаемым знаком attribution edge. Это не заменяет
replacement fidelity, но помогает понять, насколько графовые гипотезы
подтверждаются interventions.

## Минимальные рабочие пороги

Рабочий warning threshold был установлен как:

```text
last_token_top1_agreement >= 0.30
```

Это не научная константа и не финальная цель. Это минимальный инженерный
порог, ниже которого replacement model явно слишком часто выбирает другой
следующий токен. Для сильной replacement model желательны значения ближе к
`0.5-0.8`, но для текущего research scaffold значение около `0.30` уже
позволяет строить диагностические causal graphs с явными caveats.

## Experiment 1: Base v0

Config:

```text
configs/qwen2_5_0_5b_base_clt_v0.yaml
```

Model:

```text
Qwen/Qwen2.5-0.5B
```

Цель: получить первый технический baseline для CLT replacement на base-модели.
Этот запуск нужен был, чтобы проверить всю цепочку: сбор MLP activations,
обучение CLT, hook-based replacement, proxy attribution graph и causal
feature interventions.

Основные параметры:

```text
seq_len: 128
features_per_layer: 256
batch_size_sequences: 2
gradient_accumulation_steps: 8
lr: 0.0003
lambda_sparsity: 0.00005
max_optimizer_steps: 1500
```

Результаты replacement eval:

```text
top1_agreement: 0.1533
last_token_top1_agreement: 0.1834
kl_div: 5.5139
last_token_kl_div: 5.8315
mean_abs_logit_diff: 2.4691
last_token_mean_abs_logit_diff: 2.3794
target_logit_diff_mae: 1.2036
```

Causal validation proxy graph:

```text
validated nodes: 24
comparable nodes: 24
proxy/causal Pearson: 0.5968
ablation sign match rate: 0.9375
```

Интерпретация: v0 подтвердил, что pipeline технически работает, но fidelity
слишком низкая. `last_token_top1_agreement = 0.1834` означает, что replacement
model часто выбирает другой следующий токен. Поэтому attribution graph v0
нельзя трактовать как faithful circuit. Его корректный статус: proxy graph,
генерирующий гипотезы для interventions.

Важный вывод: даже при слабой replacement fidelity causal validation показал,
что часть feature-гипотез имеет реальный logit effect. Это оправдало развитие
не только proxy attribution, но и causal validation / Deep Trace.

## Experiment 2: Base v1 Fidelity Push

Config:

```text
configs/qwen2_5_0_5b_base_clt_fidelity_v1.yaml
```

Цель: проверить, достаточно ли простого увеличения capacity и числа шагов,
чтобы перейти от технического baseline к рабочей replacement model.

Основные изменения относительно v0:

```text
features_per_layer: 256 -> 512
max_optimizer_steps: 1500 -> 3000
lambda_sparsity: 0.00005
batch_size_sequences: 2
gradient_accumulation_steps: 8
```

Результаты replacement eval:

```text
top1_agreement: 0.2736
last_token_top1_agreement: 0.3185
kl_div: 3.2057
last_token_kl_div: 4.0255
mean_abs_logit_diff: 1.8484
last_token_mean_abs_logit_diff: 1.8559
target_logit_diff_mae: 0.8980
```

Интерпретация: v1 впервые прошёл минимальный warning threshold по
`last_token_top1_agreement >= 0.30`. Это показало, что проблема v0 была
частично связана с недостаточной capacity. При этом KL всё ещё высокий, а
target direction сохраняется недостаточно точно.

Важный вывод: простое увеличение CLT capacity помогает, но не даёт quality
level, достаточный для уверенного circuit tracing. Нужен более сильный
baseline.

## Experiment 3: Base v2 Main Fidelity Baseline

Config:

```text
configs/qwen2_5_0_5b_base_clt_fidelity_v2.yaml
```

Цель: получить основной base baseline для последующего Deep Trace. Этот запуск
должен был улучшить не только top1, но и KL/target direction fidelity.

Основные изменения относительно v1:

```text
features_per_layer: 512 -> 1024
max_optimizer_steps: 3000 -> 10000
lambda_sparsity: 0.00005 -> 0.00002
batch_size_sequences: 16
gradient_accumulation_steps: 1
lr: 0.0003
```

Результаты replacement eval:

```text
top1_agreement: 0.3588
last_token_top1_agreement: 0.2915
kl_div: 2.0033
last_token_kl_div: 2.0068
logit_mse: 4.7050
last_token_logit_mse: 4.3058
mean_abs_logit_diff: 1.6436
last_token_mean_abs_logit_diff: 1.5898
target_logit_diff_mae: 0.6271
target_logit_diff_mse: 0.7509
target_logit_diff_original_mean: 1.9140
target_logit_diff_replacement_mean: 1.7595
```

Интерпретация: v2 стал лучшим сбалансированным checkpoint. Он чуть не дошёл до
`0.30` по last-token top1, но существенно улучшил KL, logit diff и
`target_logit_diff_mae`. Для интерпретируемости это важнее, чем небольшой
проигрыш v1 по last-token top1, потому что target direction и форма
распределения logits сохраняются лучше.

Важный вывод: v2 выбран основным baseline для Deep Trace stage 1. Это не
идеальная replacement model, но она достаточно стабильна, чтобы строить
диагностические graphs с явным предупреждением о fidelity.

## Experiment 4: Base v3 Activation/Target Normalization

Config:

```text
configs/qwen2_5_0_5b_base_clt_fidelity_v3.yaml
```

Цель: проверить, улучшит ли layerwise normalization реконструкцию activations и
target direction. Гипотеза была такая: если CLT страдает от разных масштабов
по слоям, нормализация inputs/targets может снизить KL и стабилизировать
обучение.

Основные изменения относительно v2:

```text
features_per_layer: 1024
max_optimizer_steps: 10000
lambda_sparsity: 0.00002
normalization.enabled: true
normalize_inputs: true
normalize_targets: true
```

Фрагмент поздних training logs:

```text
optimizer_step=9800 loss=0.3286 nmse=0.3286 l0=415.26 logit_distill=none
optimizer_step=9900 loss=0.3110 nmse=0.3110 l0=413.12 logit_distill=none
optimizer_step=10000 loss=0.4055 nmse=0.4055 l0=397.59 logit_distill=none
```

Результаты replacement eval:

```text
top1_agreement: 0.3272
last_token_top1_agreement: 0.2625
kl_div: 2.1962
last_token_kl_div: 1.9621
logit_mse: 4.6351
last_token_logit_mse: 3.7716
mean_abs_logit_diff: 1.6366
last_token_mean_abs_logit_diff: 1.4729
target_logit_diff_mae: 0.6622
```

Интерпретация: normalization улучшила часть continuous metrics:
`last_token_kl_div`, `last_token_logit_mse`,
`last_token_mean_abs_logit_diff`. Но она ухудшила top1 agreement и
`target_logit_diff_mae` относительно v2. То есть модель стала чуть лучше
калибрована по логитам, но хуже сохраняет именно нужный target direction и
top-token agreement.

Важный вывод: v3 полезен как ablation, но не заменяет v2 как основной baseline.

## Experiment 5: Base v4 Lightweight Last-Token Logit Distillation

Config:

```text
configs/qwen2_5_0_5b_base_clt_fidelity_v4.yaml
```

Цель: проверить, может ли lightweight distillation на last-token logits
улучшить agreement replacement model с original model. Гипотеза была, что
прямой logit-level objective поможет там, где reconstruction loss недостаточно
связан с downstream logits.

Основные параметры:

```text
features_per_layer: 1024
normalization.enabled: true
batch_size_sequences: 8
gradient_accumulation_steps: 2
lr: 0.0002
lambda_sparsity: 0.00002
max_optimizer_steps: 10000
logit_distillation.enabled: true
normalize_logits: true
```

Первый eval с `seq_len: 256`:

```text
top1_agreement: 0.3259
last_token_top1_agreement: 0.3552
kl_div: 7.4800
last_token_kl_div: 7.9803
mean_abs_logit_diff: 2.3553
last_token_mean_abs_logit_diff: 2.0358
target_logit_diff_mae: 0.7566
```

Fair eval с `seq_len: 128`:

```text
top1_agreement: 0.3212
last_token_top1_agreement: 0.3050
kl_div: 7.5444
last_token_kl_div: 8.4962
mean_abs_logit_diff: 2.3553
last_token_mean_abs_logit_diff: 2.1445
target_logit_diff_mae: 0.8526
```

Поздний улучшенный v4 eval:

```text
top1_agreement: 0.3702
last_token_top1_agreement: 0.3224
kl_div: 5.2292
last_token_kl_div: 5.6431
logit_mse: 6.0139
last_token_logit_mse: 5.0563
mean_abs_logit_diff: 1.9036
last_token_mean_abs_logit_diff: 1.7388
target_logit_diff_mae: 0.8392
```

Интерпретация: v4 улучшает top1/ranking, но сильно портит KL и target
direction. Это ожидаемый риск logit distillation: модель может научиться
поднимать правильный top token, не сохраняя корректную форму распределения и
важные logit differences.

Важный вывод: v4 не подходит как основной circuit baseline. Для задач, где
важен только next-token top1, он интересен. Для Deep Trace он хуже v2, потому
что graph должен объяснять logit direction, а не только top-token match.

## Experiment 6: Instruct v2

Config:

```text
configs/qwen2_5_0_5b_instruct_clt_fidelity_v2.yaml
```

Model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Цель: проверить переносимость подхода на instruct-модель, которая ближе к
будущей прикладной/domain-tuned модели. В instruct config включён chat template,
и replacement/deep trace wrapper форматирует prompt через template до
tokenization.

Основные параметры:

```text
features_per_layer: 1024
max_optimizer_steps: 10000
lambda_sparsity: 0.00002
batch_size_sequences: 16
gradient_accumulation_steps: 1
chat_template: true
```

Результаты replacement eval:

```text
top1_agreement: 0.3637
last_token_top1_agreement: 0.0843
kl_div: 3.1431
last_token_kl_div: 5.1889
logit_mse: 5.9678
last_token_logit_mse: 8.9866
mean_abs_logit_diff: 1.7194
last_token_mean_abs_logit_diff: 2.3402
target_logit_diff_mae: 0.9801
```

Deep Trace single prompt:

```text
nodes: 169
edges: 293
trace kind: deep_trace_stage1
causal sign match rate: 0.875
```

Интерпретация: общий `top1_agreement` выглядит сопоставимо с base v2, но
`last_token_top1_agreement = 0.0843` очень низкий. Для causal prompts это
критично, потому что именно last-token position определяет следующий токен и
target direction. Поэтому Instruct v2 пока можно использовать только как
диагностический branch, но не как основной baseline для выводов.

Важный вывод: instruct-модель требует отдельной настройки обучения. Нельзя
механически переносить base v2 параметры и ожидать такой же last-token
fidelity.

## Experiment 7: Base v2 Deep Trace Stage 1

Checkpoint:

```text
outputs/base_clt_fidelity_v2/clt_step_10000.pt
```

Цель: перейти от proxy attribution к более глубокой graph architecture. Deep
Trace stage 1 должен добавить typed nodes, replacement errors, residual stream
nodes, layernorm nodes, attention output nodes и causal ablation edges.

Важно: `stage 1` здесь означает стадию tracing architecture, а не checkpoint
`base_clt_fidelity_v1`. Основной checkpoint для Deep Trace stage 1 - base v2.

Single-prompt result:

```text
prompt: Demand is greater than generation, so the price will
nodes: 169
edges: 268
node types:
  AttentionHeadNode
  CLTFeatureNode
  LayerNormNode
  LogitTargetNode
  MLPErrorNode
  ResidualStreamNode
causal sign match rate: 0.8333
mean MLP error norm: 7.2550
```

Top MLP error nodes:

```text
L20 error_norm=15.7945 target_projection=0.2459
L19 error_norm=10.4158 target_projection=-0.1788
L21 error_norm=18.5371 target_projection=-0.1674
L18 error_norm=7.7861  target_projection=-0.1625
L23 error_norm=25.7932 target_projection=-0.0696
L22 error_norm=22.6266 target_projection=-0.0612
```

Top causal feature edges:

```text
L19:F137 causal_effect=-0.2500
L23:F232 causal_effect=0.1875
L23:F913 causal_effect=-0.1250
L23:F976 causal_effect=-0.1250
L0:F278  causal_effect=-0.0625
L20:F487 causal_effect=0.0625
```

Prompt suite result:

```text
prompts: 5
mean sign_match: 0.750
median sign_match: 0.833
mean MLP error norm: 7.817
```

Per-prompt sign match:

```text
1. Demand is greater than generation...             0.833
2. Demand is lower than supply...                   0.667
3. Fuel supply falls while demand stays high...     0.875
4. Generation exceeds demand...                     0.875
5. A product becomes scarce while buyers need it... 0.500
```

Top error layers across suite:

```text
L20 mean_abs_proj=0.2400 mean_error_norm=17.510
L21 mean_abs_proj=0.1924 mean_error_norm=20.159
L19 mean_abs_proj=0.1553 mean_error_norm=11.569
L18 mean_abs_proj=0.0917 mean_error_norm=8.911
L23 mean_abs_proj=0.0859 mean_error_norm=29.135
L22 mean_abs_proj=0.0767 mean_error_norm=24.415
```

Интерпретация: Deep Trace v2 выявил устойчивый late-layer bottleneck:
ошибки replacement model в `L18-L23`, особенно `L20`, `L21`, `L19`, заметно
влияют на target direction. Это стало основанием для следующего эксперимента с
late-layer weighted reconstruction loss.

Важный вывод: base v2 является лучшим текущим baseline, но его Deep Trace надо
читать вместе с error nodes. Часть graph edges объясняет не исходную Qwen, а
остаточную ошибку CLT replacement.

## Experiment 8: Base Late-Layer Loss v1

Config:

```text
configs/qwen2_5_0_5b_base_clt_late_loss_v1.yaml
```

Цель: точечно улучшить late layers, потому что Deep Trace suite на v2 показал,
что самые важные replacement errors находятся в `L18-L23`. Это не
произвольный тюнинг: он следует из диагностического графа.

Основные изменения относительно base v2:

```text
features_per_layer: 1024
max_optimizer_steps: 10000
lambda_sparsity: 0.00002
batch_size_sequences: 32
lr: 0.0003
layer_loss_weights:
  L18: 2.0
  L19: 2.0
  L20: 3.0
  L21: 3.0
  L22: 2.0
  L23: 2.0
```

Training behavior:

```text
loss decreased from about 12 to about 0.28-0.31
nmse decreased from about 14 to about 0.31
l0 decreased from about 513 to about 112
logit_distill=none
```

Результаты replacement eval:

```text
top1_agreement: 0.3740
last_token_top1_agreement: 0.2954
kl_div: 1.8986
last_token_kl_div: 2.0308
logit_mse: 4.6603
last_token_logit_mse: 3.8016
mean_abs_logit_diff: 1.6266
last_token_mean_abs_logit_diff: 1.4889
target_logit_diff_mae: 0.6634
```

Deep Trace changes versus base v2:

```text
mean MLP error norm improved: 7.255 -> 6.996 on the main prompt
L20 target projection improved: 0.2459 -> 0.190
L21 target projection improved: 0.1674 -> 0.143
L18 target projection improved: 0.1625 -> 0.142
L22 target projection improved: 0.0612 -> 0.044
L23 became a larger bottleneck: about 0.0696 -> 0.245
```

Prompt suite:

```text
sign_match values: 0.833, 1.000, 0.667, 1.000, 0.833
mean sign_match: about 0.867
```

Интерпретация: late-layer weighted loss улучшил общую replacement fidelity:
`top1_agreement`, KL и mean abs logit diff стали лучше, а causal sign
stability выросла. Но target-specific metric ухудшилась:
`target_logit_diff_mae = 0.6634` против `0.6271` у base v2. Кроме того,
исправление одних поздних слоёв сделало `L23` главным residual bottleneck.

Важный вывод: точечное усиление поздних слоёв оправдано, но layer weighting
сам по себе не гарантирует улучшения target direction. Нужен более аккуратный
target-aware objective.

## Experiment 9: Base Late-Layer Target Loss v1 Attempt

Config:

```text
configs/qwen2_5_0_5b_base_clt_late_target_v1.yaml
```

Цель: совместить late-layer weighted reconstruction с target-aware loss по
направлению `" increase" - " decrease"`. Этот эксперимент был ответом на два
наблюдения: late_loss_v1 улучшил sign stability, но ухудшил target MAE; Deep
Trace показал, что `L23` остаётся bottleneck.

После проверки `metrics.jsonl` выяснилось, что этот cloud run нельзя считать
валидным target-loss экспериментом:

```text
target_logit_diff_loss: None
```

Причина: в облаке запускалась старая копия `train_clt.py`, которая сохраняла
новый YAML в checkpoint, но не использовала блок
`training.target_logit_diff_loss` в training loop. Поэтому результаты ниже
надо интерпретировать как L23-upweighted late-layer reconstruction run, а не
как доказательство влияния target-aware objective.

Основные изменения относительно late_loss_v1:

```text
batch_size_sequences: 16
lr: 0.00025
L23 layer_loss_weight: 4.0
target_logit_diff_loss.enabled: true
target_logit_diff_loss.weight: 0.05
target_logit_diff_loss.loss_type: smooth_l1
positive: " increase"
negative: " decrease"
target_pos: -1
```

Фактический training log:

```text
step=1     target_logit_diff_loss: None
step=10000 target_logit_diff_loss: None
```

Результаты replacement eval:

```text
top1_agreement: 0.3558
last_token_top1_agreement: 0.2722
kl_div: 2.0412
last_token_kl_div: 1.8848
logit_mse: 4.8209
last_token_logit_mse: 3.8790
mean_abs_logit_diff: 1.6671
last_token_mean_abs_logit_diff: 1.5070
target_logit_diff_mae: 0.6820
target_logit_diff_mse: 0.8159
target_logit_diff_original_mean: 1.9140
target_logit_diff_replacement_mean: 1.7843
```

Deep Trace single prompt:

```text
nodes: 169
edges: 251
causal sign match rate: 1.0
```

Deep Trace prompt suite:

```text
prompt 1 sign_match: 1.0 mean_error_norm=7.410
prompt 2 sign_match: 1.0 mean_error_norm=8.015
prompt 3 sign_match: 1.0 mean_error_norm=8.071
prompt 4 sign_match: 1.0 mean_error_norm=7.856
prompt 5 sign_match: 1.0 mean_error_norm=7.958
```

Интерпретация: causal sign consistency стала идеальной на suite, но это нельзя
приписывать target-aware loss. Более вероятное объяснение - сочетание
изменённых late-layer weights, повышенного веса `L23`, другого learning rate и
того же seed. Replacement fidelity при этом хуже base v2 и late_loss_v1, а
`target_logit_diff_mae` тоже хуже.

Важный вывод: old `late_target_v1` нельзя использовать как результат
target-aware обучения. Его можно хранить только как диагностический
L23-upweighted reconstruction run и как пример того, почему в training logs
обязательно нужно проверять `target_diff=...`, а не только наличие YAML-поля в
checkpoint.

## Experiment 10: Base Late-Layer Target Loss v2

Config:

```text
configs/qwen2_5_0_5b_base_clt_late_target_v2.yaml
```

Цель: повторить late_target setup с более слабым target-aware objective.
Гипотеза была такой: если вес `0.05` слишком велик, то уменьшение до `0.02`
сохранит часть target-direction пользы и меньше повредит общей replacement
fidelity.

Изменение относительно late_target_v1:

```text
target_logit_diff_loss.weight: 0.05 -> 0.02
output_dir: outputs/base_clt_late_target_v2
```

Фактический результат replacement eval:

```text
top1_agreement: 0.0011
last_token_top1_agreement: 0.0000
kl_div: 14.9466
last_token_kl_div: 14.1860
logit_mse: 30.2246
last_token_logit_mse: 29.4853
mean_abs_logit_diff: 4.4761
last_token_mean_abs_logit_diff: 4.4360
target_logit_diff_mae: 0.6170
target_logit_diff_mse: 0.7402
target_logit_diff_original_mean: 1.9140
target_logit_diff_replacement_mean: 1.8697
```

Сравнение с основным baseline `base v2`:

```text
base v2 target_logit_diff_mae: 0.6271
v2 target-loss target_logit_diff_mae: 0.6170

base v2 last_token_top1_agreement: 0.2915
v2 target-loss last_token_top1_agreement: 0.0000

base v2 kl_div: 2.0033
v2 target-loss kl_div: 14.9466
```

Интерпретация: target-aware loss действительно слегка улучшил целевую
метрику `target_logit_diff_mae`, но цена оказалась неприемлемой: replacement
model почти полностью потеряла общий logit landscape. Такой checkpoint нельзя
использовать для Deep Trace, потому что он будет объяснять не исходную Qwen и
не faithful replacement model, а деградировавшую модель, оптимизированную под
один scalar direction.

Важный вывод: single-direction target logit supervision не подходит как
достаточно сильный auxiliary objective в текущем виде. Даже `weight: 0.02`
оказался слишком агрессивным, если применять его в полном replacement forward
на каждом micro-step.

## Experiment 11: Base Late-Layer Tiny Target Loss v3

Config:

```text
configs/qwen2_5_0_5b_base_clt_late_target_v3.yaml
```

Цель: проверить, может ли target-aware objective работать только как очень
слабый регуляризатор, а не как заметная движущая сила обучения.

Изменение относительно late_target_v2:

```text
target_logit_diff_loss.weight: 0.02 -> 0.001
target_logit_diff_loss.every_n_micro_steps: 1 -> 10
output_dir: outputs/base_clt_late_target_v3
```

Гипотеза: если проблема v2 была в слишком сильном и слишком частом
single-direction градиенте, то tiny target-loss должен сохранить fidelity
примерно на уровне late-layer reconstruction run и дать лишь небольшой bias в
сторону target direction.

Фактический результат replacement eval:

```text
top1_agreement: 0.0748
last_token_top1_agreement: 0.0116
kl_div: 5.3412
last_token_kl_div: 5.5657
logit_mse: 9.1344
last_token_logit_mse: 9.5493
mean_abs_logit_diff: 2.3751
last_token_mean_abs_logit_diff: 2.4632
target_logit_diff_mae: 0.9751
target_logit_diff_mse: 1.5609
target_logit_diff_original_mean: 1.9140
target_logit_diff_replacement_mean: 1.8749
```

Сравнение с `base v2`:

```text
base v2 top1_agreement: 0.3588
late_target_v3 top1_agreement: 0.0748

base v2 last_token_top1_agreement: 0.2915
late_target_v3 last_token_top1_agreement: 0.0116

base v2 kl_div: 2.0033
late_target_v3 kl_div: 5.3412

base v2 target_logit_diff_mae: 0.6271
late_target_v3 target_logit_diff_mae: 0.9751
```

Интерпретация: даже очень слабый и редкий target-loss не оказался безопасным.
Он уже не разрушил модель так катастрофически, как `late_target_v2`, но всё
равно сильно ухудшил global replacement fidelity и одновременно ухудшил
target-direction MAE. Это закрывает single-direction target-loss как
практичный training objective для текущей CLT setup.

Важный вывод: target direction нужно оставить как evaluation/validation
metric, а не использовать как training loss. Дальше следует улучшать
reconstruction fidelity общими методами.

## Planned Experiment 12: Base Reconstruction Fidelity v1

Config:

```text
configs/qwen2_5_0_5b_base_clt_recon_fidelity_v1.yaml
```

Цель: вернуться от target-specific objectives к общей fidelity replacement
model. Этот эксперимент не использует target-aware loss, logit distillation
или normalization. Он проверяет более простой путь: больше CLT capacity,
мягче sparsity, больше данных и больше optimizer steps.

Изменение относительно `base_clt_fidelity_v2`:

```text
features_per_layer: 1024 -> 1536
max_optimizer_steps: 10000 -> 15000
max_train_tokens: 25000000 -> 50000000
lambda_sparsity: 0.00002 -> 0.00001
lr: 0.0003 -> 0.0002
target_logit_diff_loss: disabled
logit_distillation: disabled
normalization: disabled
```

Гипотеза: если основная проблема v2 - нехватка capacity и слишком жёсткая
sparsity, то larger sparse reconstruction model должна улучшить KL/logit MSE
без разрушения top1 и без переоптимизации одного target direction.

Критерии успеха:

```text
kl_div < 2.0
last_token_kl_div <= 2.0
top1_agreement >= 0.36
last_token_top1_agreement >= 0.30
mean_abs_logit_diff < 1.64
target_logit_diff_mae <= 0.627
```

Если этот run улучшит global fidelity, его стоит использовать как новый
основной replacement baseline и строить Deep Trace suite уже на нём. Если
качество не улучшится, следующий общий шаг - не target loss, а ещё более
крупная capacity или architectural optimization вроде chunked triangular
decoder / sparse operations.

## Сводная таблица

| Experiment | Main idea | top1 | last top1 | KL | last KL | target MAE | Main conclusion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| base v0 | first scaffold baseline | 0.153 | 0.183 | 5.514 | 5.831 | 1.204 | too weak, proxy only |
| base v1 | more capacity, 3000 steps | 0.274 | 0.319 | 3.206 | 4.026 | 0.898 | passes minimal top1 threshold |
| base v2 | 1024 features, 10000 steps | 0.359 | 0.292 | 2.003 | 2.007 | 0.627 | best balanced baseline |
| base v3 | activation/target normalization | 0.327 | 0.263 | 2.196 | 1.962 | 0.662 | better continuous last-token calibration, worse top1/target |
| base v4 | last-token logit distillation | 0.370 | 0.322 | 5.229 | 5.643 | 0.839 | better ranking, worse distribution |
| instruct v2 | instruct model transfer | 0.364 | 0.084 | 3.143 | 5.189 | 0.980 | not reliable at last token |
| late_loss v1 | weighted late-layer reconstruction | 0.374 | 0.295 | 1.899 | 2.031 | 0.663 | improves global fidelity/sign stability, not target MAE |
| late_target v1 attempt | L23-upweighted run; target loss inactive | 0.356 | 0.272 | 2.041 | 1.885 | 0.682 | invalid as target-loss result |
| late_target v2 | target loss 0.02 every step | 0.001 | 0.000 | 14.947 | 14.186 | 0.617 | target MAE improves slightly, replacement collapses |
| late_target v3 | target loss 0.001 every 10 steps | 0.075 | 0.012 | 5.341 | 5.566 | 0.975 | tiny target loss still hurts fidelity |

## Текущий выбор baseline

Основной baseline для исследования:

```text
base_clt_fidelity_v2
```

Причина: это лучший баланс между KL, target direction fidelity и usable
replacement behavior. Он не идеален, но на текущем этапе лучше всего подходит
для Deep Trace stage 1.

Лучший diagnostic run для sign stability:

```text
base_clt_late_target_v1
```

Причина: он показывает идеальный causal sign match на prompt suite, но old run
не считал target-loss. Поэтому это диагностический L23-upweighted result, а не
доказательство пользы target-aware objective.

Лучший late-layer improvement run:

```text
base_clt_late_loss_v1
```

Причина: он улучшает глобальную fidelity и снижает часть late-layer errors, но
не решает target MAE и создаёт/усиливает L23 bottleneck.

## Главные проблемы, выявленные экспериментами

Первая проблема: replacement fidelity всё ещё ниже уровня, достаточного для
полноценного circuit tracing. Даже лучший base v2 имеет
`last_token_top1_agreement = 0.2915`, то есть находится около минимального
инженерного порога.

Вторая проблема: разные objectives улучшают разные аспекты fidelity. v4
улучшает top1, но портит KL. v3 улучшает часть continuous metrics, но портит
top1. Old `late_target_v1` улучшил causal sign match, но target-loss в нём
фактически не работал. Валидные `late_target_v2` и `late_target_v3` показали,
что single-direction target-aware objective не является безопасным training
loss: он либо катастрофически рушит replacement fidelity, либо ухудшает и
global fidelity, и target MAE.

Третья проблема: late-layer errors остаются ключевым bottleneck. Deep Trace
suite стабильно указывает на `L18-L23`, особенно `L20`, `L21`, `L19` и затем
`L23`.

Четвёртая проблема: Instruct branch пока не готов для основных выводов.
Несмотря на нормальный all-token top1, last-token fidelity очень низкая.

Пятая проблема: Deep Trace stage 1 всё ещё не является full path attribution.
Он глубже proxy attribution, но attention пока представлен layer-level output,
а не полноценной per-head/path decomposition.

## Рекомендуемое направление дальше

Ближайший эксперимент:

```text
base_clt_recon_fidelity_v1
```

Его цель - улучшать replacement fidelity общими методами: увеличить CLT
capacity, ослабить sparsity, дать больше данных и больше optimizer steps, но
не добавлять target-specific losses.

Если этот run не улучшит KL/top1, следующий путь - не усиливать target loss
дальше, а продолжать общую fidelity-ветку:

```text
more capacity
less aggressive sparsity
late-layer capacity allocation
better replacement-conditioned training
per-head attention decomposition for Deep Trace
larger activation cache with memmap/safetensors
```

Для финальной НИР формулировка должна быть аккуратной: текущий проект уже
переходит от proxy attribution к Deep Trace stage 1, но ещё не достигает
уровня полноценного production-grade circuit tracing.
