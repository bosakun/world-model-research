# Latent-Action Token Dynamics

Status: completed mechanism study on 2026-08-23. Next-token dynamics work, but inferred action codes do not cleanly recover true controls; not a Genie reproduction.

## Purpose

Infer a discrete intervention code from adjacent visual-token grids without action labels, then condition categorical future-token prediction and interactive rollout on that code.

## Problem

Internet/video data often lacks action labels. Interactive world models need a controllable bottleneck that explains change, but unsupervised latent actions are identifiable only up to permutation—and may capture factors other than human controls.

## Previous Model

`01_vq_video_tokenizer` produces 4×4 discrete frame tokens but has no temporal/action model.

## Hypothesis

A five-way hard Gumbel bottleneck and balanced usage can improve changed-token prediction. Best-permutation action accuracy may remain low because coarse tokens/background allow alternative partitions.

## Architecture

```text
(tokens_t,tokens_t+1) -> embeddings/MLP -> 5 logits -> hard Gumbel latent action
tokens_t + latent action + spatial positions -> 2-layer Transformer -> logits_t+1 [4,4,32]

initial tokens + supplied latent action sequence -> argmax tokens -> VQ decoder -> video
```

## Data Flow

The frozen Phase 10/01 tokenizer produces pairs. True actions are stored only for evaluation. Training jointly learns action inference and spatial token dynamics using weighted categorical loss, per-example confident codes, and batch-marginal diversity.

## Tensor Shapes

Current/next IDs `[B,4,4]`; embedded pair `[B,2*16*32]`; action logits/one-hot `[B,5]`; dynamics tokens `[B,16,64]`; categorical output `[B,4,4,32]`; rollout `[B,5,4,4]`.

## Mathematics

```text
q_phi(a|x_t,x_t+1)=Categorical(logits_phi)
a_st=hard_gumbel_softmax(logits,tau=0.5)
p_theta(x_t+1|x_t,a_st)=Categorical(Transformer_theta(...)).
```

```text
L_token = weighted CE; weight=6 for changed IDs, 1 otherwise
L_conf  = mean H(q_phi(a|pair))
L_bal   = sum_a q_bar(a) log q_bar(a) = -H(q_bar)
L=L_token+0.03L_conf+0.1L_bal.
```

Minimizing confidence entropy makes each assignment sharp; minimizing negative marginal entropy encourages batch-wide code use. Neither forces semantic alignment with true actions.

## Code Mapping

- checkpoint-compatible frozen tokenizer: `tokenizer.py`
- unlabeled token pairs: `dataset.py::TokenTransitionDataset`
- latent inference/Gumbel bottleneck: `model.py::infer_action`
- action-conditioned Transformer: `model.py::predict`
- interactive token feedback: `model.py::rollout`
- weighted/change and entropy objectives: `train.py::objective`
- permutation matching/baselines: `evaluate.py`

## Training

Seed 223; `vq-moving-square-pairs-v2`; 512/128 six-frame videos; 2-pixel moves; Adam `7e-4`; batch 128 pairs; 50 epochs/1,000 steps; 238,501 parameters; checkpoint includes frozen tokenizer and dynamics.

## Losses

Changed-token weighting prevents static background from dominating. Token CE trains future prediction. Confidence avoids soft ambiguous actions. Balance avoids one-code collapse. There is deliberately no true-action classification loss.

## Evaluation Interface

`python 10_video_world_model/02_latent_action_dynamics/evaluate.py` reports best-permutation true-action accuracy, overall/changed/copy token metrics, code usage, rollout accuracy, and decoded video.

## Smoke Test Results

Five tests passed. Overall next-token accuracy `0.830`; changed-token accuracy `0.787`; copy-current baseline `0.777`; changed fraction `0.223`. All five codes were used, but best-permutation true-action accuracy was only `0.270` (chance `0.2`). Rollout token accuracy declined from `1.0` to `0.563` by frame 6.

## Failure Cases

- Initial 1-pixel/unweighted run achieved `0.893` overall accuracy but only `0.253` action alignment by copying background/static tokens.
- Larger motion and changed-token weighting improved dynamic tokens but not semantic action identifiability.
- Inferred codes can partition changes by position/boundary/visual pattern instead of direction.
- Argmax token rollout compounds errors and loses uncertainty.
- VQ codebook itself uses only five visual codes.

## Findings

Balanced, sharp latent codes are not automatically “actions.” Action labels used only after training reveal this distinction. Overall token accuracy is misleading without copy and changed-token baselines.

## Limitations

Pair-only inference, no long temporal context, tiny videos, deterministic moves, no action prior, no stochastic future, no massive unsupervised data, and no joint tokenizer training. These differ radically from Genie.

## Compare Later

Supervised action upper bound; longer context; continuous/discrete action counts; token resolution; changed weighting; code mutual information; intervention consistency; rollout accuracy/perplexity; stochastic sampling.

## Final Model Candidate

```text
Candidate: No in current form.
Reason: Tokens are controllable mechanically but latent codes do not reliably correspond to true interventions.
Advantages: action labels unnecessary; categorical interface; interactive rollout.
Disadvantages: non-identifiability; background shortcuts; compounding argmax error.
Possible conflicts: downstream policies require stable action semantics.
```

## Next Questions

Which temporal/data constraints make latent actions intervention-consistent? How can heterogeneous observations and commands condition one simulator without confusing modalities?

## References

### Genie: Generative Interactive Environments

- Authors: Jake Bruce et al.
- Year: 2024
- Paper: https://arxiv.org/abs/2402.15391
- Used for: latent action model, discrete visual-token dynamics, interactive autoregressive rollout.
- Implementation: `model.py`, `train.py`, `evaluate.py`.

Classification: **Simplified educational implementation**. Architecture, data, losses, and scale are independent reductions; observed action alignment is not a Genie result.
