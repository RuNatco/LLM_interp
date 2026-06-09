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

In managed notebook environments, do not prepend `site-packages` to
`PYTHONPATH`. The scripts already prefer the local `src` directory.

Optional Hugging Face cache variable:

```bash
export HF_HOME=.cache/huggingface
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

By default this reads `outputs/base_clt_v0/replacement_eval_metrics.json` and
prints fidelity warnings before building the proxy graph. Use
`--strict-fidelity` when you want low-fidelity replacement to stop the run.

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
  --validation-report outputs/base_clt_v0/price_graph_validated.json \
  --rank 1 \
  --select-by abs_causal_effect \
  --value 0.0 \
  --positive " increase" \
  --negative " decrease"
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

By default this reads `outputs/instruct_clt_v0/replacement_eval_metrics.json`
and prints fidelity warnings before building the proxy graph.

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
  --validation-report outputs/instruct_clt_v0/price_graph_validated.json \
  --rank 1 \
  --select-by abs_causal_effect \
  --value 0.0 \
  --positive " increase" \
  --negative " decrease"
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

Inspect these first:

```text
metrics.last_token_top1_agreement
metrics.last_token_kl_div
metrics.target_logit_diff_mae
diagnostics.layer_*
diagnostics.prefix_0_to_*
```

If full replacement is poor but single-layer replacement is acceptable, use
single-layer causal analysis first. If prefix metrics collapse after a specific
layer, inspect that layer's reconstruction NMSE and consider more CLT capacity.

Causal validation:

```bash
cat outputs/base_clt_v0/price_graph_validated.json
cat outputs/instruct_clt_v0/price_graph_validated.json
```

Key interpretation rule:

```text
proxy graph edges are hypotheses.
validated causal effect = logit_diff(intervened_logits) - logit_diff(replacement_logits).
use proxy_causal_pearson and ablation_sign_match_rate to judge graph reliability.
```

## 4. Base Fidelity Push

Use this run when the base replacement is below the `0.30` last-token top1
warning threshold and you want to try a stronger CLT without overwriting `v0`.

Train the larger CLT:

```bash
python scripts/01_train_clt.py \
  --config configs/qwen2_5_0_5b_base_clt_fidelity_v1.yaml
```

Evaluate replacement fidelity:

```bash
python scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_base_clt_fidelity_v1.yaml
```

Inspect:

```bash
cat outputs/base_clt_fidelity_v1/replacement_eval_metrics.json
```

Target checkpoint:

```text
metrics.last_token_top1_agreement >= 0.30
metrics.target_logit_diff_mae <= 1.0
```

If the gate passes or gets close, continue with graph and validation:

```bash
python scripts/03_build_attribution_graph.py \
  --checkpoint outputs/base_clt_fidelity_v1/clt_final.pt \
  --prompt "Demand is greater than generation, so the price will" \
  --positive " increase" \
  --negative " decrease" \
  --output outputs/base_clt_fidelity_v1/price_graph.json

python scripts/06_validate_attribution_graph.py \
  --checkpoint outputs/base_clt_fidelity_v1/clt_final.pt \
  --graph outputs/base_clt_fidelity_v1/price_graph.json \
  --prompt "Demand is greater than generation, so the price will" \
  --positive " increase" \
  --negative " decrease" \
  --top-k 24 \
  --output outputs/base_clt_fidelity_v1/price_graph_validated.json
```

Then select the strongest validated feature:

```bash
python scripts/04_run_feature_interventions.py \
  --checkpoint outputs/base_clt_fidelity_v1/clt_final.pt \
  --prompt "Demand is greater than generation, so the price will" \
  --validation-report outputs/base_clt_fidelity_v1/price_graph_validated.json \
  --rank 1 \
  --select-by abs_causal_effect \
  --value 0.0 \
  --positive " increase" \
  --negative " decrease"
```

## 5. Base Fidelity v2

Use this only after `base_clt_fidelity_v1` improves over `v0`. This run is
substantially longer and targets the `0.4-0.6` last-token top1 range.

Train:

```bash
python scripts/01_train_clt.py \
  --config configs/qwen2_5_0_5b_base_clt_fidelity_v2.yaml
```

Evaluate fast fidelity metrics:

```bash
python scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_base_clt_fidelity_v2.yaml
```

Inspect:

```bash
cat outputs/base_clt_fidelity_v2/replacement_eval_metrics.json
```

Target range:

```text
metrics.last_token_top1_agreement: 0.4-0.6
metrics.target_logit_diff_mae <= 0.7
```

If fidelity improves, build and validate the graph:

```bash
python scripts/03_build_attribution_graph.py \
  --checkpoint outputs/base_clt_fidelity_v2/clt_final.pt \
  --prompt "Demand is greater than generation, so the price will" \
  --positive " increase" \
  --negative " decrease" \
  --output outputs/base_clt_fidelity_v2/price_graph.json

python scripts/06_validate_attribution_graph.py \
  --checkpoint outputs/base_clt_fidelity_v2/clt_final.pt \
  --graph outputs/base_clt_fidelity_v2/price_graph.json \
  --prompt "Demand is greater than generation, so the price will" \
  --positive " increase" \
  --negative " decrease" \
  --top-k 24 \
  --output outputs/base_clt_fidelity_v2/price_graph_validated.json
```

If `clt_final.pt` was not written because cloud storage filled up, use the last
valid step checkpoint explicitly:

```bash
python scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_base_clt_fidelity_v2.yaml \
  --checkpoint outputs/base_clt_fidelity_v2/clt_step_10000.pt
```

Use the same `--checkpoint outputs/.../clt_step_10000.pt` override for graph
building and causal validation.

## 6. Base Fidelity v3: Activation/Target Normalization

Use this when `v2` improves KL/target direction but last-token top1 is still
near `0.3`. `v3` keeps the `v2` capacity and enables layerwise normalization
for CLT inputs and reconstruction targets.

Train:

```bash
python scripts/01_train_clt.py \
  --config configs/qwen2_5_0_5b_base_clt_fidelity_v3.yaml
```

Evaluate:

```bash
python scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_base_clt_fidelity_v3.yaml
```

Inspect:

```bash
cat outputs/base_clt_fidelity_v3/replacement_eval_metrics.json
```

Compare against `v2`:

```text
metrics.last_token_top1_agreement should not regress.
metrics.last_token_kl_div and metrics.target_logit_diff_mae should improve or hold.
```

If `v3` wins, build and validate the graph:

```bash
python scripts/03_build_attribution_graph.py \
  --checkpoint outputs/base_clt_fidelity_v3/clt_final.pt \
  --prompt "Demand is greater than generation, so the price will" \
  --positive " increase" \
  --negative " decrease" \
  --output outputs/base_clt_fidelity_v3/price_graph.json

python scripts/06_validate_attribution_graph.py \
  --checkpoint outputs/base_clt_fidelity_v3/clt_final.pt \
  --graph outputs/base_clt_fidelity_v3/price_graph.json \
  --prompt "Demand is greater than generation, so the price will" \
  --positive " increase" \
  --negative " decrease" \
  --top-k 24 \
  --output outputs/base_clt_fidelity_v3/price_graph_validated.json
```

## 7. Base Fidelity v4: Last-Token Logit Distillation

Use this after `v3`. `v4` adds a lightweight replacement-conditioned objective:
the replacement model is run inside training and its last-token top-k logits are
matched to original Qwen logits. This is slower than `v3`.

Train:

```bash
python scripts/01_train_clt.py \
  --config configs/qwen2_5_0_5b_base_clt_fidelity_v4.yaml
```

Evaluate:

```bash
python scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_base_clt_fidelity_v4.yaml
```

Inspect:

```bash
cat outputs/base_clt_fidelity_v4/replacement_eval_metrics.json
```

Primary target:

```text
metrics.last_token_top1_agreement > v3
metrics.last_token_kl_div does not regress sharply
metrics.target_logit_diff_mae <= v3
```

If GPU memory is too high, edit only the micro-batch size in the YAML:

```text
training.batch_size_sequences: 4
training.gradient_accumulation_steps: 4
```

## 8. Deep Trace Stage 1

Use `base_clt_fidelity_v2` as the main Deep Trace baseline. It has the best
current balance for target-direction fidelity and replacement distribution
fidelity.

Here `stage 1` names the tracing architecture stage, not the
`base_clt_fidelity_v1` checkpoint. The Deep Trace checkpoint baseline is v2.

```text
proxy attribution
-> validated proxy graph
-> cached replacement trace
-> typed circuit graph with error nodes
-> causal pruning
-> frontend-compatible graph
```

Build a typed Deep Trace graph on base v2:

```bash
python scripts/07_build_deep_trace_graph.py \
  --checkpoint outputs/base_clt_fidelity_v2/clt_step_10000.pt \
  --prompt "Demand is greater than generation, so the price will" \
  --positive " increase" \
  --negative " decrease" \
  --max-feature-nodes 64 \
  --top-error-nodes 8 \
  --causal-top-k 8 \
  --output outputs/base_clt_fidelity_v2/price_deep_trace.json \
  --cache-output outputs/base_clt_fidelity_v2/price_deep_trace_cache.pt
```

If `clt_final.pt` exists and is known-good, you can use it instead of
`clt_step_10000.pt`.

Inspect:

```bash
python scripts/08_summarize_deep_trace_graph.py \
  --graph outputs/base_clt_fidelity_v2/price_deep_trace.json \
  --top-n 8 \
  --output outputs/base_clt_fidelity_v2/price_deep_trace_summary.json
```

Key fields:

```text
metadata.node_types
metadata.edge_kinds
metadata.causal_pruning.sign_match_rate
metadata.cache_summary.mean_mlp_error_norm
edges[kind=causal_ablation_effect]
nodes[type=MLPErrorNode]
```

Deep Trace stage 1 is deeper than proxy attribution, but still not full path
attribution. Attention is represented as layer-level output, not per-head
decomposition.

Build a small prompt suite after the first graph works:

```bash
python scripts/09_build_deep_trace_prompt_suite.py \
  --checkpoint outputs/base_clt_fidelity_v2/clt_step_10000.pt \
  --output-dir outputs/base_clt_fidelity_v2/deep_trace_suite \
  --summary-output outputs/base_clt_fidelity_v2/deep_trace_suite_summary.json \
  --max-feature-nodes 64 \
  --top-error-nodes 8 \
  --causal-top-k 8
```

This uses five default supply/demand prompts. To use your own list, save a JSON
array of prompt strings and pass:

```bash
python scripts/09_build_deep_trace_prompt_suite.py \
  --checkpoint outputs/base_clt_fidelity_v2/clt_step_10000.pt \
  --prompts-file configs/deep_trace_prompts.json \
  --output-dir outputs/base_clt_fidelity_v2/deep_trace_suite \
  --summary-output outputs/base_clt_fidelity_v2/deep_trace_suite_summary.json
```

Caches are not saved by default for the suite. Add `--cache-dir
outputs/base_clt_fidelity_v2/deep_trace_suite_cache` only when you need the
activation tensors.

## 9. Instruct Deep Trace

Train the high-fidelity Instruct CLT after the base Deep Trace baseline is
stable:

```bash
python scripts/01_train_clt.py \
  --config configs/qwen2_5_0_5b_instruct_clt_fidelity_v2.yaml
```

Evaluate replacement fidelity:

```bash
python scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_instruct_clt_fidelity_v2.yaml
```

Build the Instruct Deep Trace graph:

```bash
python scripts/07_build_deep_trace_graph.py \
  --checkpoint outputs/instruct_clt_fidelity_v2/clt_final.pt \
  --prompt "Demand is greater than generation, so the price will" \
  --positive " increase" \
  --negative " decrease" \
  --max-feature-nodes 64 \
  --top-error-nodes 8 \
  --causal-top-k 8 \
  --output outputs/instruct_clt_fidelity_v2/price_deep_trace.json \
  --cache-output outputs/instruct_clt_fidelity_v2/price_deep_trace_cache.pt
```

The Instruct checkpoint config has `chat_template: true`, and the proxy/deep
trace wrapper applies that template before tokenization.

## 10. Base Late-Layer Loss

Use this after the base v2 Deep Trace suite shows late-layer replacement errors
in `L18-L23`. This run keeps the base v2 CLT capacity but gives larger
reconstruction-loss weight to late layers:

```text
L18, L19, L22, L23: weight 2.0
L20, L21:           weight 3.0
```

Train:

```bash
python scripts/01_train_clt.py \
  --config configs/qwen2_5_0_5b_base_clt_late_loss_v1.yaml
```

Evaluate replacement fidelity:

```bash
python scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_base_clt_late_loss_v1.yaml
```

Build the single-prompt Deep Trace:

```bash
python scripts/07_build_deep_trace_graph.py \
  --checkpoint outputs/base_clt_late_loss_v1/clt_final.pt \
  --prompt "Demand is greater than generation, so the price will" \
  --positive " increase" \
  --negative " decrease" \
  --max-feature-nodes 64 \
  --top-error-nodes 8 \
  --causal-top-k 8 \
  --output outputs/base_clt_late_loss_v1/price_deep_trace.json \
  --cache-output outputs/base_clt_late_loss_v1/price_deep_trace_cache.pt
```

Summarize:

```bash
python scripts/08_summarize_deep_trace_graph.py \
  --graph outputs/base_clt_late_loss_v1/price_deep_trace.json \
  --top-n 8 \
  --output outputs/base_clt_late_loss_v1/price_deep_trace_summary.json
```

Run the prompt suite:

```bash
python scripts/09_build_deep_trace_prompt_suite.py \
  --checkpoint outputs/base_clt_late_loss_v1/clt_final.pt \
  --output-dir outputs/base_clt_late_loss_v1/deep_trace_suite \
  --summary-output outputs/base_clt_late_loss_v1/deep_trace_suite_summary.json \
  --max-feature-nodes 64 \
  --top-error-nodes 8 \
  --causal-top-k 8
```

Compare against base v2:

```text
target_logit_diff_mae should move below 0.627.
last_token_kl_div should stay near 2-3.
last_token_top1_agreement should be >= 0.30 or at least not regress.
top MLPErrorNode projections in L18-L23 should decrease.
Deep Trace suite mean/median sign_match should improve or hold.
```

## 11. Base Late-Layer Target Loss

Use this after `base_clt_late_loss_v1`. That run improved Deep Trace suite
stability but did not improve `target_logit_diff_mae`, and it exposed `L23` as
a remaining replacement-error bottleneck.

This run combines:

```text
late-layer weighted reconstruction
L23 weight 4.0
target-aware loss on logit(" increase") - logit(" decrease")
no broad top-k distillation
```

Train:

```bash
python scripts/01_train_clt.py \
  --config configs/qwen2_5_0_5b_base_clt_late_target_v1.yaml
```

Evaluate:

```bash
python scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_base_clt_late_target_v1.yaml
```

Build the single-prompt Deep Trace:

```bash
python scripts/07_build_deep_trace_graph.py \
  --checkpoint outputs/base_clt_late_target_v1/clt_final.pt \
  --prompt "Demand is greater than generation, so the price will" \
  --positive " increase" \
  --negative " decrease" \
  --max-feature-nodes 64 \
  --top-error-nodes 8 \
  --causal-top-k 8 \
  --output outputs/base_clt_late_target_v1/price_deep_trace.json \
  --cache-output outputs/base_clt_late_target_v1/price_deep_trace_cache.pt
```

Summarize:

```bash
python scripts/08_summarize_deep_trace_graph.py \
  --graph outputs/base_clt_late_target_v1/price_deep_trace.json \
  --top-n 8 \
  --output outputs/base_clt_late_target_v1/price_deep_trace_summary.json
```

Run the prompt suite:

```bash
python scripts/09_build_deep_trace_prompt_suite.py \
  --checkpoint outputs/base_clt_late_target_v1/clt_final.pt \
  --output-dir outputs/base_clt_late_target_v1/deep_trace_suite \
  --summary-output outputs/base_clt_late_target_v1/deep_trace_suite_summary.json \
  --max-feature-nodes 64 \
  --top-error-nodes 8 \
  --causal-top-k 8
```

Compare against both base v2 and late_loss_v1:

```text
target_logit_diff_mae should improve below 0.627 if the target-aware loss works.
last_token_kl_div should stay near 2-3.
last_token_top1_agreement should cross or approach 0.30.
L23 replacement_error_projection should decrease versus late_loss_v1.
Deep Trace suite mean sign_match should stay near or above late_loss_v1.
```

## 12. Base Late-Layer Tiny Target Loss

Use this after `base_clt_late_target_v2` if `weight: 0.02` improves
`target_logit_diff_mae` but collapses global replacement fidelity. This run
keeps the same late-layer weighting, but makes target supervision a very weak
regularizer:

```text
target_logit_diff_loss.weight: 0.001
target_logit_diff_loss.every_n_micro_steps: 10
```

Train:

```bash
python scripts/01_train_clt.py \
  --config configs/qwen2_5_0_5b_base_clt_late_target_v3.yaml
```

Evaluate:

```bash
python scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_base_clt_late_target_v3.yaml
```

Build the single-prompt Deep Trace only if replacement fidelity does not
collapse:

```bash
python scripts/07_build_deep_trace_graph.py \
  --checkpoint outputs/base_clt_late_target_v3/clt_final.pt \
  --prompt "Demand is greater than generation, so the price will" \
  --positive " increase" \
  --negative " decrease" \
  --max-feature-nodes 64 \
  --top-error-nodes 8 \
  --causal-top-k 8 \
  --output outputs/base_clt_late_target_v3/price_deep_trace.json \
  --cache-output outputs/base_clt_late_target_v3/price_deep_trace_cache.pt
```

Compare against base v2 and late_target_v2:

```text
target_logit_diff_mae should improve or hold near 0.627.
kl_div should stay near 2-3, not jump toward late_target_v2's ~15.
last_token_top1_agreement should stay near 0.28-0.30 or better.
top1_agreement should not collapse below 0.30.
```

## 13. Base Reconstruction Fidelity v1

Use this after target-loss runs show that single-direction logit supervision
hurts replacement fidelity. This run returns to general reconstruction quality:
more CLT capacity, softer sparsity, more data, more optimizer steps, and no
target-aware/logit-distillation objectives.

Train:

```bash
python scripts/01_train_clt.py \
  --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v1.yaml
```

Evaluate:

```bash
python scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v1.yaml
```

Inspect:

```bash
cat outputs/base_clt_recon_fidelity_v1/replacement_eval_metrics.json
```

Primary comparison targets:

```text
Compare against base_clt_fidelity_v2:
  kl_div:                    below 2.003
  last_token_kl_div:         at or below 2.007
  top1_agreement:            above 0.359
  last_token_top1_agreement: above 0.292, ideally above 0.30
  target_logit_diff_mae:     at or below 0.627
```

Build Deep Trace only if replacement fidelity improves or at least holds:

```bash
python scripts/07_build_deep_trace_graph.py \
  --checkpoint outputs/base_clt_recon_fidelity_v1/clt_final.pt \
  --prompt "Demand is greater than generation, so the price will" \
  --positive " increase" \
  --negative " decrease" \
  --max-feature-nodes 64 \
  --top-error-nodes 8 \
  --causal-top-k 8 \
  --output outputs/base_clt_recon_fidelity_v1/price_deep_trace.json \
  --cache-output outputs/base_clt_recon_fidelity_v1/price_deep_trace_cache.pt
```

Then run the prompt suite:

```bash
python scripts/09_build_deep_trace_prompt_suite.py \
  --checkpoint outputs/base_clt_recon_fidelity_v1/clt_final.pt \
  --output-dir outputs/base_clt_recon_fidelity_v1/deep_trace_suite \
  --summary-output outputs/base_clt_recon_fidelity_v1/deep_trace_suite_summary.json \
  --max-feature-nodes 64 \
  --top-error-nodes 8 \
  --causal-top-k 8
```

## 14. Base Reconstruction Fidelity v2

Use this after `base_clt_recon_fidelity_v1` improves over `base_clt_fidelity_v2`.
This run keeps the same general reconstruction objective and increases CLT
capacity:

```text
features_per_layer: 1536 -> 2048
```

Train:

```bash
python scripts/01_train_clt.py \
  --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2.yaml
```

Evaluate:

```bash
python scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2.yaml
```

Inspect:

```bash
cat outputs/base_clt_recon_fidelity_v2/replacement_eval_metrics.json
```

Primary comparison targets:

```text
Compare against base_clt_recon_fidelity_v1:
  kl_div:                    below 1.581
  last_token_kl_div:         below 1.560
  top1_agreement:            above 0.421
  last_token_top1_agreement: above 0.332
  target_logit_diff_mae:     below 0.597
```

Build Deep Trace on this checkpoint as the current best baseline:

```bash
python scripts/09_build_deep_trace_prompt_suite.py \
  --checkpoint outputs/base_clt_recon_fidelity_v2/clt_final.pt \
  --output-dir outputs/base_clt_recon_fidelity_v2/deep_trace_suite \
  --summary-output outputs/base_clt_recon_fidelity_v2/deep_trace_suite_summary.json \
  --max-feature-nodes 64 \
  --top-error-nodes 8 \
  --causal-top-k 8
```

## 15. Continue Reconstruction Fidelity v2

Use this when `base_clt_recon_fidelity_v2` is the current best checkpoint and
you want to continue training from its weights without overwriting the original
output directory. This is an initialization-from-checkpoint fine-tune, not a
full optimizer-state resume.

Train for 5000 additional optimizer steps with a smaller learning rate:

```bash
python scripts/01_train_clt.py \
  --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v1.yaml
```

Evaluate:

```bash
python scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v1.yaml
```

Inspect:

```bash
cat outputs/base_clt_recon_fidelity_v2_continue_v1/replacement_eval_metrics.json
```

Compare against `base_clt_recon_fidelity_v2`:

```text
kl_div should stay below 1.496 or improve.
last_token_kl_div should stay below 1.427 or improve.
last_token_top1_agreement should stay above 0.369 or improve.
target_logit_diff_mae should stay below 0.579 or improve.
```

## 16. Continue Reconstruction Fidelity v2 Again

Use this after `base_clt_recon_fidelity_v2_continue_v1` improves fidelity. This
run starts from the continued checkpoint, lowers the learning rate again, and
uses a shorter budget to reduce overfitting risk.

Train:

```bash
python scripts/01_train_clt.py \
  --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v2.yaml
```

Evaluate:

```bash
python scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v2.yaml
```

Inspect:

```bash
cat outputs/base_clt_recon_fidelity_v2_continue_v2/replacement_eval_metrics.json
```

Compare against `base_clt_recon_fidelity_v2_continue_v1`:

```text
kl_div should stay below 1.380 or improve.
last_token_kl_div should stay below 1.228 or improve.
last_token_top1_agreement should stay above 0.541 or improve.
target_logit_diff_mae should stay below 0.541 or improve.
last_token_logit_mse should not regress much further.
```
