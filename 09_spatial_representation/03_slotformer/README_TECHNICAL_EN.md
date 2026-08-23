# SlotFormer Temporal Slot Dynamics

Status: completed on 2026-08-22. Simplified educational temporal-slot Transformer using ground-truth ordered slots; not a full SlotFormer reproduction.

## Purpose

Predict multiple object slots jointly across time with causal self-attention and measure the teacher-forcing/autoregressive rollout gap.

## Problem

Object binding alone does not model motion. Recurrent flat states compress history, while slot dynamics should preserve entities and let all objects in a frame interact when predicting future frames.

## Previous Model

C-SWM predicts one step from object slots and actions. The Slot Attention smoke run did not produce reliable masks. Feeding those masks forward would conflate perception failure with temporal modeling.

## Hypothesis

Given reliable ordered position slots, a small frame-causal Transformer should infer velocity/interaction from four context frames, but autoregressive error should accumulate over eight future frames.

## Architecture

```text
positions [B,T,K=2,2]
 -> slot projection + time embedding + slot-ID embedding
 -> flatten time-major [B,T*K,64]
 -> 2-layer, 4-head Transformer with frame-causal mask
 -> residual next-slot prediction [B,T,K,2]
 -> append last predicted frame and repeat for rollout
```

## Data Flow

Two objects have hidden velocity, bounce at boundaries, and exchange equal/opposite close-range impulses. Only positions are input, so history is needed. During training each observed frame predicts the next. During evaluation four real frames seed an eight-frame autoregressive rollout.

## Tensor Shapes

Sequences `[B,12,2,2]`; training input/target `[B,11,2,2]`; tokens `[B,22,64]`; mask `[22,22]`; context `[B,4,2,2]`; rollout `[B,8,2,2]`.

## Mathematics

For token `(t,k)`:

```text
x_tk = W_s s_tk + e_time(t) + e_slot(k)
M[(t,k),(u,j)] = blocked iff u > t
s_hat_(t+1,k) = s_tk + W_o Transformer_M(x)_tk.
```

Thus all slots at time `t` attend one another, but no query sees a future frame. Training uses `mean ||s_hat_(t+1)-s_(t+1)||²`. Rollout feeds predictions back, changing the input distribution and accumulating errors.

## Code Mapping

- hidden-velocity relational sequences: `dataset.py`
- block causal mask: `model.py::frame_causal_mask`
- tokenization/Transformer/residual head: `model.py::SlotFormer.forward`
- autoregressive feedback: `model.py::SlotFormer.rollout`
- horizon diagnostics: `evaluate.py`

## Training

Seed 179; `relational-slot-sequences-v1`; 768/192 sequences; Adam `5e-4`; batch 64; 60 epochs/720 steps; 101,570 parameters; no dropout; checkpoint format 1.

## Losses

Teacher-forced next-position MSE teaches temporal and relational prediction at every observed frame. There is no reconstruction, contrastive, matching, or multi-step training loss in this isolated experiment.

## Evaluation Interface

`python 09_spatial_representation/03_slotformer/evaluate.py` writes per-future-frame RMSE and a two-object rollout plot.

## Smoke Test Results

Five tests passed, including exact causal invariance to future edits. Validation teacher-forced MSE was `0.001207`. Autoregressive position RMSE rose from `0.0303` at horizon 1 to `0.2682` at horizon 8.

## Failure Cases

- Teacher-forced accuracy understates closed-loop rollout error.
- Slot IDs are assumed stable and explicitly embedded; permutation/matching is not solved.
- Boundary bounce and close interaction are rare modes that amplify errors.
- Predicted positions can leave the physical support because the network has no hard bounds.

## Findings

Frame-level causal masking is materially different from a simple token-triangular mask: same-frame slots need bidirectional interaction. Even with perfect slots, autoregressive prediction still suffers compounding error.

## Limitations

Ground-truth positions replace learned visual slots; ordering is fixed; there is no Slot Attention encoder, decoder, stochastic dynamics, action conditioning, or long-video benchmark. This isolates only temporal slot attention.

## Compare Later

Compare GRU per slot, flat Transformer, relational graph, and SlotFormer under matched states. Ablate same-frame attention, slot IDs, history length, multi-step loss, and scheduled sampling. Measure per-object/horizon error, collision/bounce subsets, permutation stability, latency, and memory.

## Final Model Candidate

```text
Candidate: Undecided
Reason: Joint temporal slot modeling works with reliable slots, but binding and long rollout remain unresolved.
Advantages: explicit entities; direct long history access; same-frame relational attention.
Disadvantages: quadratic token cost; fixed slot identity here; compounding rollout error.
Possible conflicts: unordered visual slots require matching or permutation-equivariant temporal treatment.
```

## Next Questions

How should learned visual slots be aligned across frames? Can multi-step objectives or stochastic slot dynamics reduce the horizon-8 error?

## References

### SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models

- Authors: Ziyi Wu, Nikita Dvornik, Klaus Greff, Thomas Kipf, Animesh Garg
- Year: 2022
- Paper: https://arxiv.org/abs/2210.05861
- Used for: Transformer dynamics over object-centric slots and autoregressive future-slot prediction.
- Implementation: `model.py`, `train.py`, `evaluate.py`.

Classification: **Simplified educational implementation**. Ground-truth ordered position slots and synthetic physics are independent modifications; the paper's full visual pipeline is not reproduced.
