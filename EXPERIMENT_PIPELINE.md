# Experiment pipeline

Run these commands from the project root:

```bash
cd qwen_clt_circuit_baseline
```

## 0. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt
pip install -e .
```

## 1. Base CLT

Train CLT:

```bash
python scripts/01_train_clt.py \
  --config configs/qwen2_5_0_5b_base_clt_v0.yaml
```

Evaluate CLT replacement logits:

```bash
python scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_base_clt_v0.yaml
```

Build proxy attribution graph:

```bash
python scripts/03_build_attribution_graph.py \
  --checkpoint outputs/base_clt_v0/clt_final.pt \
  --prompt "Demand is greater than generation, so the price will" \
  --positive " increase" \
  --negative " decrease" \
  --output outputs/base_clt_v0/price_graph.json
```

Validate graph nodes causally:

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

Visualize graph:

```bash
python scripts/05_visualize_attribution_graph_svg.py \
  --graph outputs/base_clt_v0/price_graph.json \
  --prompt "Demand is greater than generation, so the price will" \
  --output outputs/base_clt_v0/price_graph_viz.svg
```

Optional single-feature intervention check:

```bash
python scripts/04_run_feature_interventions.py \
  --checkpoint outputs/base_clt_v0/clt_final.pt \
  --prompt "Demand is greater than generation, so the price will" \
  --layer 20 \
  --pos 9 \
  --feature 126 \
  --value 0.0
```

## 2. Instruct CLT

Train CLT:

```bash
python scripts/01_train_clt.py \
  --config configs/qwen2_5_0_5b_instruct_clt_v0.yaml
```

Evaluate CLT replacement logits:

```bash
python scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_instruct_clt_v0.yaml
```

Build proxy attribution graph:

```bash
python scripts/03_build_attribution_graph.py \
  --checkpoint outputs/instruct_clt_v0/clt_final.pt \
  --prompt "Demand is greater than generation, so the price will" \
  --positive " increase" \
  --negative " decrease" \
  --output outputs/instruct_clt_v0/price_graph.json
```

Validate graph nodes causally:

```bash
python scripts/06_validate_attribution_graph.py \
  --checkpoint outputs/instruct_clt_v0/clt_final.pt \
  --graph outputs/instruct_clt_v0/price_graph.json \
  --prompt "Demand is greater than generation, so the price will" \
  --positive " increase" \
  --negative " decrease" \
  --top-k 24 \
  --output outputs/instruct_clt_v0/price_graph_validated.json
```

Visualize graph:

```bash
python scripts/05_visualize_attribution_graph_svg.py \
  --graph outputs/instruct_clt_v0/price_graph.json \
  --prompt "Demand is greater than generation, so the price will" \
  --output outputs/instruct_clt_v0/price_graph_viz.svg
```

Optional single-feature intervention check:

```bash
python scripts/04_run_feature_interventions.py \
  --checkpoint outputs/instruct_clt_v0/clt_final.pt \
  --prompt "Demand is greater than generation, so the price will" \
  --layer 23 \
  --pos 9 \
  --feature 80 \
  --value 0.0
```

## 3. What to inspect

Training:

```bash
tail -n 5 outputs/base_clt_v0/metrics.jsonl
tail -n 5 outputs/instruct_clt_v0/metrics.jsonl
```

Replacement evaluation:

```bash
cat outputs/base_clt_v0/replacement_eval_metrics.json
cat outputs/instruct_clt_v0/replacement_eval_metrics.json
```

Causal validation:

```bash
cat outputs/base_clt_v0/price_graph_validated.json
cat outputs/instruct_clt_v0/price_graph_validated.json
```

Key interpretation rule:

```text
proxy graph edges are hypotheses.
validated causal effect = logit_diff(intervened_logits) - logit_diff(replacement_logits).
```
