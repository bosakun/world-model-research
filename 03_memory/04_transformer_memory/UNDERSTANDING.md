# Understanding Transformer Memory

## What problem does this solve?

Recurrent models can only consult the past through the current hidden state. Transformer memory keeps a window of past tokens and lets each prediction retrieve a weighted mixture from that explicit history. The central research question is whether direct access helps when an important observation occurred many steps ago.

## Before

GRU/RSSM memory follows a chain:

```text
h_0 -> h_1 -> h_2 -> ... -> h_t
```

Every update must preserve old information that may become relevant later. The chain has fixed-size storage and sequential computation.

## After

Transformer memory stores:

```text
x_0, x_1, ..., x_t
```

and computes a context for token `t` by attending to tokens `0...t`. It still has finite memory—`max_context` bounds retained tokens—but it does not force all history into one recurrent vector.

## Core Idea

Each time step becomes one token containing the current visual latent and the action to execute. A learned position vector says where the token occurs. Causal self-attention lets the prediction at time `t` choose among present and past tokens while mathematically excluding future tokens.

Attention is dynamic addressing, not perfect symbolic recall. The network still has to learn useful queries, keys, values, token representations, and prediction losses.

## Data Flow

```text
o_t -> Encoder -> z_t
                   + a_t
                     |
              content token
                     + position p_t
                     |
               causal Transformer
                     |
                context c_t
                     |
             predict z_{t+1}
                     |
        Decoder and Goal-state head
```

Teacher forcing uses true encoded `z_t` at every token. Autoregressive rollout starts from `z_0`, then feeds each `z_hat_{t+1}` back as the next input. This difference is why both metrics are necessary.

## Mathematics

### Latent/action tokenization

```text
x_t = W_x [z_t,a_t] + p_t
```

- `z_t`: 16-dimensional visual latent for the current observation.
- `a_t`: four-dimensional one-hot action taken after that observation.
- `W_x`: projection from 20 content dimensions to model width 64.
- `p_t`: learned absolute position embedding.

Why needed: attention operates on fixed-width tokens. Action conditioning is required to distinguish futures caused by different controls. Position is required because attention alone is permutation-equivariant and does not know temporal order.

### Queries, keys, and values

```text
Q=XW_Q, K=XW_K, V=XW_V
```

- query: what information this prediction seeks;
- key: how each stored token advertises its content;
- value: information retrieved if that token is weighted.

Why needed: separate learned projections let retrieval depend on relationships between the current need and stored history.

### Scaled dot-product attention

```text
A = softmax((QK^T)/sqrt(d_k) + M)
Y = AV
```

- `A_ij`: weight from query position `i` to key position `j`.
- `sqrt(d_k)`: controls dot-product scale so softmax does not saturate simply because vectors are wide.
- `M`: causal mask.

Why needed: it produces a differentiable, content-dependent summary of stored tokens.

### Causal mask

```text
M_ij = 0          if j <= i
M_ij = -infinity  if j > i
```

After softmax, future positions have exactly zero attention weight. Why needed: without it, training can read `z_{t+1}` or later tokens and report a prediction that uses the answer.

### Multi-head attention

```text
MultiHead(X) = Concat(head_1,...,head_H) W_O
```

Why needed: different heads can learn different retrieval relationships. Head diversity is possible, not guaranteed; the evaluation plot averages heads only for a compact visualization.

### Residual block

```text
Y = X + MHA(LN(X), M)
C = Y + FFN(LN(Y))
```

Why needed: residual paths preserve token information and ease optimization; layer normalization controls activation scale; the position-wise FFN transforms retrieved features nonlinearly.

### Next-state prediction

```text
z_hat_{t+1} = W_o c_t
```

Why needed: attention itself only builds a context. A prediction head maps that context back to the latent state consumed by the decoder and next rollout step.

### Training loss

```text
L = L_reconstruct(o_t, Decoder(z_t))
  + L_future(o_{t+1}, Decoder(z_hat_{t+1}))
  + 0.5 MSE(z_hat_{t+1}, stopgrad(z_{t+1}))
  + 0.1 CE(goal_hat_{t+1}, goal_{t+1}).
```

Why needed: reconstruction makes the latent decodable; future-image loss teaches consequences; latent loss stabilizes transition learning; semantic loss prevents the tiny Goal from disappearing in a good-looking aggregate MSE.

## Tensor Shapes

```text
observations:       [B,7,3,20,20]
actions:            [B,6,4]
latents:            [B,7,16]
token content:      [B,6,20]
tokens/context:     [B,6,64]
Q/K/V per head:     [B,4,6,16]
attention maps:     [2,B,4,6,6]
predicted latents:  [B,6,16]
predicted images:   [B,6,3,20,20]
```

## Code Mapping

| Concept | Code |
|---|---|
| `o_t -> z_t` | `transformer_memory.py::VisualEncoder` |
| content and position tokens | `TransformerMemoryDynamics.tokenize` |
| causal `M` | `TransformerMemoryDynamics.causal_mask` |
| Q/K/V multi-head attention | `CausalTransformerBlock.attention` |
| residual/norm/FFN | `CausalTransformerBlock.forward` |
| `c_t -> z_hat_{t+1}` | `prediction_head` |
| teacher-forced sequence | `TransformerMemoryDynamics.forward` |
| prediction-fed rollout | `TransformerMemoryDynamics.rollout` |
| image/semantic objectives | `transformer_losses.py` |
| attention visualization | `evaluate.py::evaluate` |

## Important Components

### Action in every token

Why necessary: world dynamics are action-conditioned. Omitting action forces the same state token to predict an average over all controls.

### Positional embedding

Why necessary: identical content at two times otherwise has no order label. Tests check that identical zero content receives different token embeddings at different positions.

### Causal attention

Why necessary: training has the full sequence in memory. The mask enforces the information boundary that deployment/rollout faces. Tests change future tokens by a huge amount and verify earlier outputs remain identical.

### Explicit context window

Why necessary: it defines what “memory” means operationally. Tokens older than `max_context` are not accessible and rollout uses a sliding window when necessary.

### Autoregressive rollout

Why necessary: teacher-forced one-step prediction does not expose compounding error. Real imagination must consume model-created states.

## What happens if we remove it?

- Remove action: different controls from the same image become ambiguous.
- Remove position: the model receives a bag of latent/action pairs rather than an ordered trajectory.
- Remove causal mask: one-step metrics can become invalid through future leakage.
- Remove history and keep only the last token: the architecture becomes a memory-free feed-forward transition.
- Remove residual paths/norm: optimization can become less stable as depth increases.
- Remove image/semantic targets: latent MSE can collapse with a jointly moving encoder unless another task anchors meaning.
- Replace teacher-forced inputs with predicted latents during all training: exposure mismatch may shrink, but early prediction errors can make optimization much harder.
- Never test autoregressive rollout: compounding errors remain hidden.

## Recurrent memory versus attention memory

| Question | GRU/RSSM recurrent path | Causal Transformer |
|---|---|---|
| Stored form | one fixed-size hidden vector | bounded list of tokens |
| Access old event | only through all intermediate updates | direct attention edge within context |
| Teacher-forced parallelism | sequential recurrent update | all causal positions in parallel |
| Rollout state | hidden vector | token history/cache |
| Main scaling | roughly linear in sequence length | dense attention roughly quadratic |
| Order | recurrence itself | explicit position information |

## What I Should Be Able to Explain

- What exactly is stored in one token?
- Why is action `a_t` paired with latent `z_t` to predict `z_{t+1}`?
- Why does self-attention need positional information?
- Which entries of the causal mask are forbidden, and why?
- Why is changing future input a strong causal-mask test?
- What are query, key, and value in intuitive terms?
- What does one attention-map row mean?
- Why are attention weights not proof of causal importance?
- What is different between teacher forcing and autoregressive rollout?
- Why can rollout pixel error be low while decoded Goal traces are visibly imperfect?
- How is this different from TransDreamer's complete TSSM?
- What would KV caching change, and what would it not change?

## Questions

- At what history length does direct attention become measurably better than a 64-dimensional recurrent state?
- Do particular heads specialize in the first visible Goal token?
- Would relative or rotary positions generalize beyond the trained context more naturally?
- Should stochastic state be tokenized, predicted distributionally, or inferred with a separate posterior?
- Can scheduled sampling or block teacher forcing reduce autoregressive drift without destabilizing training?
- How should sparse/object tokens replace one vector per frame in later phases?
