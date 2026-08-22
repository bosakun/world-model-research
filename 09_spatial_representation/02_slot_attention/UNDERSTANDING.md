# Understanding Slot Attention

## What problem does this solve?

It attempts to bind an unordered set of image features into a smaller unordered set of entity-like representations without being told a pixel-to-object assignment.

## Before

C-SWM's object axis came from a fixed channel split. Binding was preprocessing, not learning.

## After

Every spatial token competes across slots, and each slot repeatedly updates from the tokens for which it takes responsibility. A shared decoder tests whether their composition explains the image.

## Core Idea

Slots are exchangeable queries. Attention is unusual because softmax is over slots, not input tokens: this creates competition for each token. A second normalization lets each slot aggregate its assigned evidence.

## Data Flow

`pixels -> CNN tokens + positions -> sampled slots -> [attention, weighted update, GRU, MLP] x3 -> slot RGB/masks -> mixture reconstruction`.

## Mathematics

`a_nk=softmax_k(k(x_n)^Tq(s_k)/sqrt(D))`: token `n` divides responsibility among slots `k`; required for competition.

`w_nk=a_nk/sum_n a_nk`, `u_k=sum_n w_nk v(x_n)`: each slot receives a normalized token summary; required so update scale does not grow with assigned area.

`s_k'=GRU(u_k,s_k)+MLP(LN(...))`: preserves previous hypothesis while refining it over iterations.

`x_hat=sum_k softmax_k(alpha_k) rgb_k`: independently decoded slots compose through masks; required to link slots back to image evidence.

## Code Mapping

The exact competition is `SlotAttention.forward`; `GRUCell` performs recurrent refinement; `SlotAttentionAutoencoder` tokenizes and decodes; `best_permutation_iou` handles arbitrary slot order.

## Important Components

Random distinct initialization breaks slot symmetry. Positional features retain location. Slot-axis softmax creates competition. Recurrent iterations allow grouping hypotheses to improve. Shared decoder prevents fixed slot-index parameters. Permutation-aware metrics respect exchangeability.

## What happens if we remove it?

- Competition: all slots can attend identically.
- Random asymmetry: identically initialized slots remain identical under shared computation.
- Re-normalization: large regions dominate update magnitude.
- GRU/iterations: grouping is a single feed-forward guess.
- Positional embedding: identical local appearance loses location.
- Mask softmax: decoded layers do not form a normalized composition.
- Permutation matching: correct but swapped slots score as wrong.
- Foreground weighting here: sparse-background blank reconstruction becomes cheap.

## What I Should Be Able to Explain

- Along which axis does Slot Attention apply softmax, and why?
- Why are slot identities permutation-equivariant?
- Why can low reconstruction error coexist with poor object masks?
- What did foreground weighting fix, and what did it fail to fix?
- Why is a fixed RNG seed needed for reproducible sampled slots?
- Why should failed Slot Attention masks not be passed uncritically into temporal dynamics?

## Questions

- Would a convolutional broadcast decoder induce better locality?
- Should temporal matching or motion cues be part of the binding loss?
- Which regularizers improve specialization without imposing wrong objects?
