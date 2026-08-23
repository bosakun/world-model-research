# Understanding Action-Conditioned JEPA

## What problem does this solve?

It learns what part of a future observation is predictable under an action without reconstructing every sensor detail.

## Before / After / Core Idea

Reconstruction asks for the whole observation. JEPA predicts a target representation. An EMA target changes slowly; action input distinguishes interventions; variance/covariance terms resist trivial agreement.

## Data Flow

`o_t -> online z_t + action -> predicted future z`; separately `o_t+1 -> EMA target z`; align them, then update the target slowly.

## Mathematics

`z_hat=p(e(o),a)` needs action because identical current states can have different controlled futures. `L_pred` aligns future features. `L_var` keeps batch dimensions active. `L_cov` discourages duplicated dimensions. `target <- EMA(online)` stabilizes targets without backpropagating into them.

## Code Mapping

`encoder` is online representation; `predictor` is action dynamics; `target_encoder` is EMA branch; `jepa_loss` implements three losses; `update_target` performs EMA; evaluation uses a state probe only after training.

## Important Components

Stop-gradient prevents target chasing within one update. EMA slows feedback. Action gives controllability. Anti-collapse losses make constant agreement costly. Linear probing tests retained physical variables without having trained them directly.

## What happens if we remove it?

- Action: zero-action ablation shows worse RMSE.
- EMA: both branches move simultaneously and targets destabilize.
- Variance: constant features can align.
- Covariance: many dimensions can duplicate one signal.
- Probe: low JEPA loss gives no evidence of physical meaning.
- Decoder absence reversed: model may allocate capacity to nuisance noise.

## What I Should Be Able to Explain

- Why can frozen target parameters still change by EMA?
- Why does stop-gradient not guarantee non-collapse?
- What do variance and covariance penalize differently?
- Why is target probe much better than predicted probe?
- What does the zero-action comparison prove?

## Questions

- Which variance target fits bounded `tanh`-free latents?
- Should target normalization or predictor depth change?
- How does uncertainty interact with JEPA alignment?
