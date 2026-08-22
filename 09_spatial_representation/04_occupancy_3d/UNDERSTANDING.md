# Understanding 3D Occupancy World Models

## What problem does this solve?

It predicts where matter occupies a 3D coordinate system rather than compressing a scene only into an uninterpreted vector or object list.

## Before

2D slots represented object attributes but not explicit volume. Collision/free-space queries required a decoder or additional geometry.

## After

Each voxel has an occupancy probability, and actions update a compact latent whose decoder produces future 3D fields.

## Core Idea

Use dense geometry as a supervised prediction target while keeping recurrent computation in a lower-dimensional latent. Evaluate both overlap and horizon degradation.

## Data Flow

`true center/action -> voxel renderer -> Conv3D encoder -> latent action transition -> ConvTranspose3D -> probability grid -> IoU/center`.

## Mathematics

`p(x)=sigmoid(logit(x))` gives occupied probability for cell `x`.

Positive-weighted BCE makes rare occupied errors costly. Dice compares overlap relative to predicted/true occupied mass. Neither alone guarantees correct center.

`c_hat=sum_x p(x)x/sum_x p(x)` is differentiable center of mass. Its supervised MSE encodes geometric location, but assumes one meaningful center.

Latent consistency `||z_hat-sg(e(O_next))||²` aligns free rollout states with observed-state representations.

## Code Mapping

`render_voxels` defines occupancy; `encoder_conv`/`decoder_conv` implement 3D perception/generation; `next_latent` is action dynamics; `occupancy_loss` is BCE/Dice; `occupancy_center` is geometry supervision.

## Important Components

3D convolutions encode locality; action conditioning makes prediction controllable; positive weighting addresses sparsity; Dice addresses overlap; consistency anchors recursive latents; center loss corrects spatial blur.

## What happens if we remove it?

- Action: different commanded futures collapse together.
- 3D convolution: local volume structure becomes expensive to learn densely.
- Positive weight: all-free prediction is attractive.
- Dice: overlap receives weak relative emphasis.
- Consistency: latent drift is supervised only through a difficult sparse decoder.
- Center loss: observed smoke run fell to final IoU 0.111 with broad predictions.
- Decoder: occupancy cannot be inspected, though planning could still use latent heads.

## What I Should Be Able to Explain

- Why is occupancy different from an object list?
- Why is all-free prediction a dangerous baseline?
- What do BCE and Dice each teach?
- Why is center loss privileged and limited?
- Why does dense-grid memory scale cubically?
- Why does horizon IoU decay despite exact action input?

## Questions

- How should unknown/occluded voxels differ from free voxels?
- Can sparse voxel/octree representations replace dense grids?
- How do semantic occupancy and ego trajectory interact?
