# Understanding Latent Actions

## What problem does this solve?

It tries to infer “what intervention happened?” from before/after visual states when no action metadata is available.

## Before

Video tokens described appearance but could not be commanded. Supervised planners assumed action labels.

## After

A discrete bottleneck encodes pairwise change, and token dynamics can be driven by a supplied code. Whether the code equals a human action remains an empirical question.

## Core Idea

Force future prediction through a small discrete change variable. Use straight-through Gumbel sampling for gradients, confidence to make assignments discrete, and marginal entropy to prevent all pairs sharing one code.

## Data Flow

`frames -> frozen VQ IDs -> pair action inference -> hard code -> token Transformer -> next IDs -> recursive decoded video`.

## Mathematics

Gumbel-softmax approximates categorical sampling while `hard=True` uses one-hot forward values and soft gradients backward.

`H(q(a|pair))` should be low for confident assignments; `H(mean_pair q)` should be high for population diversity. These criteria define discreteness/use, not semantics.

Weighted CE emphasizes locations where `token_t != token_t+1`; otherwise background copy dominates.

## Code Mapping

`infer_action` is the bottleneck; `predict` is conditional dynamics; `objective` contains CE/confidence/balance; `best_mapping` acknowledges arbitrary action-code permutation; copy baseline checks triviality.

## Important Components

Frozen tokenizer stabilizes targets; changed-token weights expose motion; hard bottleneck creates interactive controls; balance prevents collapse; permutation evaluation respects label symmetry; rollout tests feedback.

## What happens if we remove it?

- Bottleneck: future encoder can leak unrestricted change.
- Hard sampling: control is not a discrete interface.
- Confidence: codes remain mixtures.
- Balance: all changes may use one code.
- Changed weights: background copy gives deceptively high accuracy.
- Permutation matching: arbitrary code names look incorrect.
- True-action audit: non-semantic partitions remain hidden.

## What I Should Be Able to Explain

- Why are latent action labels only meaningful up to permutation?
- Why do confidence and balance not guarantee true action recovery?
- Why was overall token accuracy misleading?
- How does hard Gumbel preserve gradients?
- Why does interactive rollout need supplied codes after pair inference training?

## Questions

- Does longer temporal context distinguish motion direction better?
- What intervention invariance losses are needed?
- Should latent actions be hierarchical or continuous?
