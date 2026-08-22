# Understanding VQ Video Tokens

## What problem does this solve?

It changes visual generation from predicting continuous pixels to predicting indices in a learned visual vocabulary.

## Before

Continuous latents had no finite symbolic target; small prediction drift could accumulate without an explicit nearest valid visual prototype.

## After

Each 4×4 location is one integer token. A decoder maps code embeddings back to pixels, and future models can use categorical cross-entropy.

## Core Idea

Learn encoder and codebook jointly, choose the nearest code non-differentiably, and use a straight-through gradient so the encoder still learns.

## Data Flow

`frame -> continuous grid -> nearest code IDs -> quantized embeddings -> reconstruction`; videos merely batch frames in this phase.

## Mathematics

`k*=argmin ||z_e-c_k||²` selects a discrete token. `z_st=z_e+sg(c_k-z_e)` separates forward value from backward gradient. Codebook loss updates `c_k`; commitment loss stops `z_e` from wandering; reconstruction keeps tokens visually meaningful.

Perplexity `exp(-sum p_k log p_k)` measures effective vocabulary use. It is 32 only for uniform use and 1 for one code.

## Code Mapping

`VectorQuantizer.forward` computes all distances, IDs, straight-through values, and VQ losses. `tokenize_video` restores time and grid axes. Evaluation computes histogram/perplexity.

## Important Components

Spatial downsampling creates a token grid; codebook discretizes it; straight-through enables encoder learning; commitment stabilizes assignments; decoder tests retained information; usage metrics detect collapse.

## What happens if we remove it?

- Quantizer: ordinary continuous autoencoder.
- Straight-through: reconstruction cannot train encoder through argmin.
- Codebook loss: embeddings do not follow encoder outputs.
- Commitment: encoder outputs may grow/change codes excessively.
- Decoder: tokens have no checked visual semantics.
- Usage metrics: five active codes could be mistaken for 32-code capacity.

## What I Should Be Able to Explain

- Why is nearest-code selection non-differentiable?
- What value/gradient does straight-through use?
- Why are codebook and commitment losses asymmetric?
- Why is perplexity different from codebook size?
- Why is this not yet a video dynamics model?

## Questions

- Would EMA code updates revive dead codes?
- How stable are token IDs across adjacent frames?
- Should tokenization be trained jointly with dynamics?
