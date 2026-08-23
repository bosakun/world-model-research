# Action-Conditioned JEPA for Noisy Robot Transitions

Status: completed on 2026-08-23. Simplified educational predictive-representation study inspired by V-JEPA 2; not a video/robot benchmark reproduction.

## Purpose

Predict the representation of a robot's next noisy observation from the current observation and motor action without reconstructing pixels/sensors.

## Problem

Physical observations contain nuisance noise. Reconstruction can spend capacity modeling it, while action-conditioned planning needs the controllable, predictable part of state. Joint-embedding prediction risks collapse.

## Previous Model

Earlier latent planners use supervised reward/value or explicit reconstruction. This experiment learns only future representation alignment plus anti-collapse regularization.

## Hypothesis

An online encoder/action predictor trained toward an EMA target encoder should preserve probeable physical state, use actions materially, and avoid constant latents when variance/covariance regularizers are present.

## Architecture

```text
observation_t [6] -> online encoder -> z_t [32] + action [2] -> predictor -> z_hat_t+1
observation_t+1 -> EMA target encoder -------------------------------> target z_t+1
                                      smooth-L1 + variance + covariance
```

There is no observation decoder.

## Data Flow

Synthetic robot state contains position/velocity; observations append nuisance noise. Dynamics include inertia and random slip. Online encoder/predictor update by gradient; target encoder updates only by EMA. A post-hoc linear probe audits physical state.

## Tensor Shapes

Observation/next observation `[B,6]`; action `[B,2]`; true state `[B,4]`; online/predicted/target latent `[B,32]`; covariance `[32,32]`.

## Mathematics

```text
z=e_theta(o_t); z_hat=p_phi(z,a_t); z+=e_bar(o_t+1)
L_pred=SmoothL1(z_hat,sg(z+))
L_var=mean_j max(0,1-sqrt(Var(z_hat_j)+eps))
L_cov=sum_(i!=j) Cov(z_hat)_ij^2 / D
theta_bar <- 0.99 theta_bar + 0.01 theta.
```

Prediction learns controllable temporal structure. Variance resists constant dimensions; covariance discourages redundant dimensions. EMA supplies a slowly moving target but does not alone mathematically forbid collapse.

## Code Mapping

Noisy/slipping robot transition: `dataset.py`; online/predictor/EMA target: `model.py::ActionJEPA`; objectives: `losses.py::jepa_loss`; EMA update: `train.py`; linear/action probes: `evaluate.py`.

## Training

Seed 271; `noisy-robot-transitions-v1`; 2,048/512 samples; Adam `8e-4`; batch 128; 60 epochs/960 steps; EMA 0.99; 9,376 parameters including frozen target; checkpoint format 1.

## Losses

Smooth L1 makes representation prediction robust to noisy residuals. Variance/covariance define non-collapse and redundancy pressure. No true state, reward, or reconstruction label enters training.

## Evaluation Interface

`evaluate.py` fits a probe on separate current latents, applies it to predicted next latents, compares real actions against zero actions, and reports latent standard deviation.

## Smoke Test Results

Four tests passed. Action-conditioned next-state probe RMSE `0.1600`; zero-action `0.1842`; target-encoder probe `0.01391`; predicted latent mean std `0.3458`. Validation prediction/variance/covariance losses: `0.00378/0.6524/0.1236`.

## Failure Cases

- Predictor is much worse than target representation, leaving substantial dynamics error.
- Variance remains below its desired unit threshold.
- EMA target can still co-collapse with online encoder.
- One-step synthetic dynamics do not test contact-rich or long-horizon robotics.

## Findings

Decoder-free prediction preserves linearly accessible physical state and action improves prediction. Anti-collapse metrics must be reported separately; low alignment loss alone would be misleading.

## Limitations

Vector observations, no video, robot hardware, demonstrations, online replay, planning, multi-step loss, or large pretrained encoder. Synthetic slip is aleatoric but no uncertainty head models it.

## Compare Later

Action/no-action, EMA rates, anti-collapse ablations, decoder baseline, multi-step prediction, noise/slip levels, probe RMSE, latent rank/std, rollout, and planning success.

## Final Model Candidate

```text
Candidate: Undecided
Reason: Action-aware probe improves, but predictor and variance quality remain weak.
Advantages: decoder-free; nuisance-tolerant target; action-conditioned.
Disadvantages: collapse risk; EMA coupling; no calibrated uncertainty.
Possible conflicts: reward objectives may reshape features away from generic physical prediction.
```

## Next Questions

Can the representation drive a robot through a bounded interface? How should demonstrations and online transitions share one replay schema?

## References

### V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

- Authors: Mido Assran et al.
- Year: 2025
- Paper: https://arxiv.org/abs/2506.09985
- Used for: action-conditioned predictive representation and decoder-free robot-planning motivation.
- Implementation: `model.py`, `losses.py`.

Classification: **Simplified educational implementation** using synthetic vector observations. It does not reproduce V-JEPA 2 pretraining, video architecture, datasets, or robot results.
