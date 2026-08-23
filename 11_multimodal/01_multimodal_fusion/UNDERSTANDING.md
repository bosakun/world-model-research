# Understanding Multimodal Fusion

## What problem does this solve?

It combines measurements that describe different aspects of one hidden physical state and remains explicit about missing evidence.

## Before

One condition schema was selected. The model could not use vision and proprioception together or distinguish missing touch from “no contact.”

## After

Each modality becomes a typed token; unavailable data becomes a learned missing token; self-attention integrates available evidence.

## Core Idea

Adapters translate shape, type embeddings preserve provenance, availability masks encode epistemic absence, and shared future losses ground all tokens in a common consequence.

## Data Flow

`raw modality -> adapter -> type/missing token -> self-attention -> fused state -> next physical/visual predictions`.

## Mathematics

`u_m=mask_m(A_m(x_m)+e_m)+(1-mask_m)r_m`. This is needed because zero may be a valid input. `f=Pool(Attention(u_1...u_M))` lets modalities condition one another. Ablation difference measures dependence, not necessarily causal importance.

## Code Mapping

Adapters are named `vision/proprio/language/touch`; `types` and `missing` represent provenance/absence; `torch.where` enforces masking; `fusion` integrates; two heads provide shared grounding.

## Important Components

Separate adapters respect modality statistics. Type embeddings prevent token-role confusion. Missing replacement blocks value leakage. Attention permits cross-modal interaction. Per-modality ablation exposes shortcuts.

## What happens if we remove it?

- Mask: missing and zero-valued evidence collide.
- Type embedding: equal-dimensional tokens lose provenance.
- Proprio: observed RMSE rises 16×.
- Language: action intention becomes ambiguous.
- Vision/touch here: little change, revealing redundancy rather than proving universal uselessness.
- Shared targets: tokens need another alignment signal.

## What I Should Be Able to Explain

- Why is missing not zero?
- What do adapters versus type embeddings do?
- Why does no-proprio evaluation count as distribution shift?
- Why can attention coexist with dominant-modality shortcuts?
- What does an ablation establish and not establish?

## Questions

- How should asynchronous timestamps enter tokens?
- Can uncertainty weight unreliable modalities?
- When should cross-attention replace self-attention?
