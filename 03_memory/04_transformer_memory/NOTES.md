# Transformer Memory Research Notes

Date: 2026-08-22. Notes for a future article; no article has been written.

## Questions before implementation

- Is “Transformer memory” really memory, or just a longer input tensor?
- Where should action live relative to observation latent?
- How can future leakage be tested rather than assumed absent?
- Will attention visualization show retrieval of the initially visible Goal?

## Predictions before implementation

- Causal teacher-forced training should be parallel over sequence positions.
- Autoregressive rollout will still be sequential and may be more expensive because context grows.
- Six steps may be too short to show a meaningful advantage over GRU.
- The exact causal mask is more important than architectural size.

## What became clearer during implementation

- Memory is the retained key/value token context. If a token is truncated, that event is forgotten regardless of model width.
- A Transformer removes recurrent compression, not the need for a state representation or a finite context policy.
- Pairing `[z_t,a_t]` creates a clear transition token: its output predicts `z_{t+1}`.
- Rollout must append predicted latents; feeding encoded future images would only evaluate teacher forcing.
- Position is a separate source of information from content and action.

## Errors and fixes

### Cross-experiment pytest collision

Error: the full Memory suite failed collection because both RSSM and Transformer folders used `test_dataset_contract.py`, which pytest imported under the same top-level module name.

Cause: independent experiment folders are not Python packages, so identical test basenames collide in one process.

Fix: rename the Transformer test to `test_transformer_dataset_contract.py`. The combined suite then passed 29 tests.

### Misleading evaluation vocabulary

Error: the first evaluation JSON called the plain autoencoder metric `posterior_autoencoder_reconstruction_mse`, copied conceptually from RSSM.

Cause: Transformer Memory has no posterior.

Fix: rename it `autoencoder_reconstruction_mse`. Vocabulary must reflect actual mechanisms, especially when writing paper comparisons.

## Results

- Default seed 29, 406,794 parameters, 160 Adam steps at `3e-3`.
- Train total: `1.794921 -> 0.043907`.
- Final validation total: `0.035018`.
- Autoencoder plain pixel MSE: `0.001481`.
- Teacher-forced one-step plain pixel MSE: `0.000964`.
- Six-step autoregressive mean pixel MSE: `0.000964`.
- Goal state-head accuracy: 100% teacher-forced and at all six rollout horizons.
- All 29 combined Memory tests passed.

These are one-seed smoke results, not comparative evidence.

## Interesting behavior

- The attention matrix has a strict lower-triangular/diagonal support; forbidden future weights are exactly zero.
- Later queries distribute weight over several earlier tokens instead of selecting one hard memory address.
- Rollout keeps a faint green trace after the true Goal disappears. The semantic head is correct, but the image decoder is not pixel-perfect.
- Horizon error is non-monotonic because different target frames have different visual difficulty and mostly static backgrounds.

## Figures for a future article

- `outputs/attention_map.png`: explain rows as queries, columns as available key history, and the zero upper triangle.
- `outputs/sequence_rollout.png`: teacher-forced seed versus predicted-latent feedback.
- `outputs/one_step_prediction.png`: separate one-step causal prediction from rollout.
- `outputs/rollout_error.png`: show why compounding error need not rise monotonically in this dataset.
- Diagram comparing one GRU arrow chain with one direct attention edge from the current query to an early token.

## Article-worthy explanations

- “GRU memory carries a notebook forward; attention memory keeps a shelf of past pages and learns which pages to open.” Note the shelf is bounded by context length.
- Demonstrate leakage by removing the causal mask as an explicit bad ablation in Phase 90, never as a legitimate model.
- Explain why attention visualization is an availability/weight diagnostic, not a causal proof.

## Unexpected or unresolved behavior

- State-head semantics are easier than sharp future image reconstruction.
- Attention to the initial Goal token is not visually dominant in the head-averaged map, despite correct hidden Goal classification. Individual heads or distributed representations may matter.
- Dense full-context recomputation is acceptable for six steps but unsuitable for planning horizons without KV caching or a more efficient design.

## Compare later

- Construct longer delayed-cue sequences; six steps are insufficient to favor either memory architecture.
- Match encoder, parameter budget, seed, splits, optimizer, and training steps across No Memory/GRU/RSSM/Transformer.
- History truncation, token shuffle, position removal, action removal, and per-head attention ablations.
- Measure teacher-forced throughput separately from autoregressive latency and peak memory.
