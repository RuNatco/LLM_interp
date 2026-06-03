# Experiment plan: base + instruct CLT baseline

## 1. две модели

### Base

`Qwen/Qwen2.5-0.5B` используется как технический контроль:

- проще next-token prediction;
- меньше влияния chat template и instruction-following;
- удобнее отлаживать hooks, reconstruction loss и CLT training.

### Instruct

`Qwen/Qwen2.5-0.5B-Instruct` используется как прикладной baseline:

- ближе к будущей domain-tuned instruct-модели;
- позволяет проверять instruction-style prompts;
- полезна для перехода к доменным задачам электроэнергетики.

## 2. Рекомендуемый порядок

```text
1. Base / tiny run:
   features_per_layer=128, seq_len=128, max_steps=1000

2. Base / v0 run:
   features_per_layer=512, seq_len=128, max_steps больше

3. Instruct / tiny run:
   features_per_layer=128, chat_template=true

4. Instruct / v0 run:
   features_per_layer=512

5. Attribution graph:
   proxy attribution graph for logit(" increase") - logit(" decrease");
   graph edges are hypotheses, not full circuit-tracing edges

6. Feature interventions:
   ablation top features from graph and compare
   intervened logits against replacement logits

7. Causal validation:
   run `scripts/06_validate_attribution_graph.py` on top-k proxy graph nodes
```

## 3. Что считать успешным запуском

Минимальный критерий:

- training loss не NaN;
- NMSE падает;
- L0 не равен 0;
- L0 не равен числу всех features;
- attribution graph сохраняется в JSON с `metadata.attribution_kind="proxy"`;
- top features можно занулить через feature intervention API;
- intervention API возвращает `replacement_logits`, `intervened_logits`
  и `intervention_logit_delta`.
- causal validation report сохраняет measured effects для top-k graph nodes.

## 4. Что улучшать после первого запуска

1. Validate in-forward replacement: проверить KL/top-k agreement и layer-wise
   ablations после замены `layer.mlp` outputs.
2. Chunked triangular decoder: иначе 512+ features/layer быстро станут тяжёлыми.
3. Evaluation metrics:
   - KL(original logits || replacement logits);
   - top-k agreement;
   - logit-difference recovery;
   - feature ablation effect via `intervened_logits - replacement_logits`.
4. Совместимость с visual frontend `circuit-tracer`.
5. Fuller attribution model:
   - attention-mediated feature-feature paths;
   - residual stream dynamics;
   - layernorm/error nodes;
   - replacement-model error terms.
6. Доменные prompts:
   - demand > generation → price increase;
   - generation > demand → price decrease;
   - congestion → nodal price spread;
   - anomaly in participant bids → suspicious behavior.
