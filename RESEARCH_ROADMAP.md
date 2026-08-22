# World Model Research Roadmap

Last updated: 2026-08-22

Statuses: `Not Started`, `Reading`, `Implementing`, `Evaluating`, `Completed`, `Needs Re-evaluation`.

## Repository baseline

- **Completed** — Step 0 repository audit: see `REPOSITORY_AUDIT.md`.
- **Needs Re-evaluation** — Previously recalled basic/visual implementations were absent from this checkout. If recovered, compare rather than overwrite.

## 01 Basic Dynamics

- **Not Started** — Fully observable state/action transition baseline (unavailable in audited checkout)
- **Not Started** — Multi-step state rollout experiment

## 02 Visual Latent

- **Not Started** — Image encoder/decoder and autoencoder baseline (unavailable in audited checkout)
- **Not Started** — Memory-free latent dynamics baseline
- **Not Started** — Latent rollout experiment

## 03 Memory

- **Completed** — `01_gru`: deterministic recurrent latent dynamics on fully observable Grid World
- **Completed** — `02_partial_observation`: paired local-view aliases establish equal observations with different hidden Goal states; model comparison deferred
- **Completed** — `03_rssm`: continuous-Gaussian deterministic/stochastic recurrent state, prior/posterior, reparameterization, KL, reconstruction, and prior-only rollout (21 combined tests passed; one-seed smoke run, not a comparison)
- **Completed** — `04_transformer_memory`: latent/action tokens, learned positions, multi-head causal attention, teacher-forced prediction, autoregressive rollout, and attention visualization (29 combined tests passed; one-seed smoke run)
- **Deferred to Phase 90** — matched No Memory / GRU / RSSM / Transformer comparison and memory ablations

### 03/01 hypothesis

Adding a GRU hidden state creates a valid mechanism for retaining trajectory history, but on a deterministic fully observable Grid World it may not reduce prediction error relative to a sufficiently expressive memory-free transition because the current observation already identifies the Markov state.

### 03/01 completion gate

- [x] Independent experiment directory and required documentation skeleton
- [x] Fully observable image Grid World and sequence dataset
- [x] Visual encoder/decoder and retained Simple Dynamics baseline
- [x] GRUCell dynamics, one-step path, and autoregressive rollout path
- [x] Shape, transition, gradient, finite-value, and dataset tests authored
- [x] Tests executed successfully (6 passed)
- [x] Small training and held-out evaluation executed
- [x] Output plots and metrics generated
- [x] Results, failures, and candidate decision recorded

## 04 Uncertainty

- **Completed** — `01_probabilistic_dynamics`: heteroscedastic diagonal-Gaussian transition, NLL, coverage, and sampled rollout
- **Completed** — `02_ensemble`: five-member bootstrap ensemble, epistemic/aleatoric decomposition, OOD disagreement, TS1 and TS-infinity particle rollout
- **Deferred to Phase 07** — uncertainty-aware CEM/MPC planning; probabilistic rollout interface is ready

## 05 Long Horizon

- **Completed** — `01_latent_overshooting`: all-start distance-5 recursive objective and 30-step compounding-error diagnostics
- **Completed** — `02_temporal_abstraction`: ordered five-action chunks and learned boundary-to-boundary macro transition

## 06 Reward and Value

- **Completed** — `01_prediction_heads`: immediate reward, Monte Carlo value, continuation probability, terminal-aware masking and calibration

## 07 Planning

- **Completed** — `01_random_shooting`: uniform continuous action-sequence search with reward + terminal-value scoring
- **Completed** — `02_cem`: iterative elite Gaussian refitting
- **Completed** — `03_mpc`: execute-first-action receding-horizon CEM loop
- **Completed** — `04_latent_planning`: decoder-free task-oriented latent dynamics/reward/value planning with learned-model CEM (four tests; one-seed smoke run)

## 08 Imagination RL

- **Completed** — `01_actor_critic`: reparameterized actor, EMA critic, λ-returns, frozen learned-world imagination, and exact-world audit (four tests; behavior policy exposed model exploitation)
- **Completed (documentation scope)** — Dreamer/DreamerV2/DreamerV3 mechanism comparison; full algorithms remain outside this educational implementation

## 09 Spatial Representation

- **Completed** — `01_cswm`: known-binding object slots, relational graph transition, contrastive energy, and post-hoc position probe
- **Completed (failed discovery result)** — `02_slot_attention`: iterative competitive binding and broadcast reconstruction work mechanically, but object mask IoU remained 0.271; retained as a documented collapse case
- **Completed** — `03_slotformer`: frame-causal temporal Transformer over reliable ordered slots with eight-frame autoregressive diagnostics
- **Completed** — `04_occupancy_3d`: action-conditioned 8³ voxel latent rollout with BCE, Dice, consistency, geometric-center losses, and six-horizon IoU

## 10 Video World Model

- **Implementing** — `01_video_tokenizer`: discrete video tokens and reconstruction
- **Not Started** — Genie / UniSim mechanism studies

## 11 Multimodal

- **Not Started** — Language-conditioned state and dynamics
- **Not Started** — Cross-modal alignment and grounding

## 12 Physical AI

- **Not Started** — Action-conditioned predictive representation
- **Not Started** — Robot and autonomous-driving transfer (V-JEPA 2, DayDreamer, OccWorld)

## 90 Evaluation

- **Not Started** — Unified No Memory / GRU / RSSM / Transformer Memory evaluation
- **Not Started** — One/5/10-step and long-horizon prediction under matched data and seeds
- **Not Started** — Uncertainty calibration, planning success, parameter/latency/memory accounting
- **Not Started** — Ablations, multiple seeds, and training stability

## 99 Integrated World Model

- **Not Started** — Select components from controlled comparisons
- **Not Started** — Integrate perception, memory, dynamics, uncertainty, imagination, reward/value, and planning
- **Not Started** — End-to-end smoke test and evidence-based component decision record
