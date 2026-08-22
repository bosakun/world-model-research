# Understanding SlotFormer Dynamics

## What problem does this solve?

It predicts how several entity representations evolve jointly through time while retaining a separate object axis.

## Before

C-SWM was one-step and Slot Attention addressed within-frame binding. Neither isolated long temporal attention over object histories.

## After

Every object-frame pair is a token. A causal Transformer can retrieve older motion evidence and model same-frame relations, then recursively predict future slot sets.

## Core Idea

Use time-major slot tokens with a block causal mask: future frames are hidden, but all objects in the current/past frames are visible. This respects temporal causality without inventing an order among simultaneous objects.

## Data Flow

`ordered position slots -> feature/time/slot embeddings -> block-causal attention -> residual next slots -> autoregressive append`.

## Mathematics

`x_tk=W s_tk+e_t+e_k` encodes state, time, and stable synthetic identity.

`M_qp=1` (blocked) only if the key frame is after the query frame. Object index within one frame does not affect visibility.

`s_hat_(t+1)=s_t+g(Transformer(x)_t)` uses a residual to express physical persistence.

`L=mean ||s_hat_(t+1)-s_(t+1)||²` is teacher forced; repeated rollout evaluates the distribution shift created by model outputs.

## Code Mapping

`frame_causal_mask` is causality; `time_embedding`/`slot_embedding` are positional information; `TransformerEncoder` is memory; `rollout` feeds predictions back; evaluation computes each horizon separately.

## Important Components

History is necessary because input omits velocity. Same-frame visibility enables relations. Causality prevents target leakage. Residual prediction exploits smooth motion. Autoregressive evaluation reveals compounding error hidden by teacher forcing.

## What happens if we remove it?

- Time embeddings: frames with similar positions become ambiguous.
- History: velocity direction cannot be inferred from one position frame.
- Same-frame attention: interactions become per-object independent.
- Causal mask: training can copy future evidence.
- Residual: network relearns absolute position rather than change.
- Autoregressive test: claimed long-horizon ability rests only on teacher inputs.
- Stable slot IDs here: swapping input order changes identity tracking; a general pipeline needs matching/equivariance.

## What I Should Be Able to Explain

- Why is ordinary triangular token masking wrong for simultaneous slots?
- Why do positions require multiple frames to infer velocity?
- Why can teacher-forced MSE be small while rollout RMSE grows?
- Which perception problem is deliberately bypassed?
- How does token count scale with frames and slots?

## Questions

- Can temporal attention itself resolve slot permutation?
- Would predicting distributions help at collisions?
- Which multi-step objective best controls compounding slot error?
