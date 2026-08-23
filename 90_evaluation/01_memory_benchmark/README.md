# Unified Partial-Observation Memory Benchmark

Status: completed on 2026-08-23. Matched three-seed comparison designed for this repository; not a paper benchmark.

## Purpose

Compare No Memory, GRU, continuous RSSM, and causal Transformer under one partial-observation dataset, loss, context, rollout, seeds, and evaluation contract.

## Problem

Earlier experiment-specific metrics, encoders, steps, and seeds were not comparable. Pixel error can reward a memory-free model that predicts the common local view while discarding an invisible Goal.

## Previous Model

Phases 03/01–04 proved each mechanism separately. Partial-observation data established visual aliases but did not perform a matched comparison.

## Hypothesis

No Memory must remain at chance on paired identical observations. Memory models should exceed chance, and resetting their memory should remove the advantage. RSSM/Transformer may be more stable than GRU under this small training budget.

## Architecture

```text
same partial images/actions/labels
 -> No Memory: MLP(z_t,a_t)
 -> GRU: GRUCell(z_t,a_t,h_t)
 -> RSSM: h_t + Gaussian prior/posterior z_t
 -> Transformer: causal history tokens (z_t,a_t)

context observations t=0..2 -> prior/autoregressive rollout t=3..12
```

Visual prediction reads a 16-D decodable state. Hidden-Goal classification reads each model's genuine memory state: GRU hidden, RSSM `h+z`, Transformer context token. No Memory reads its predicted latent.

## Data Flow

Each even/odd sequence pair has different initially visible Goals, identical actions, and bitwise-identical local observations when both Goals leave view. Models see two context transitions and then roll ten steps without future observations. Goal labels/full state are evaluation/training-head targets, never observation input.

## Tensor Shapes

Images `[B,13,3,20,20]`; actions `[B,12,4]`; context ends at t=2; predicted future `[B,10,3,20,20]`; Goal logits `[B,10,2]`; alias mask `[B,10]`. Latent 16; GRU/RSSM deterministic/Transformer model dimension 64.

## Mathematics

Joint matched objective:

```text
L = 5 MSE(o_hat_next,o_next) + MSE(o_recon,o)
  + MSE(z_hat_next,sg(z_next)) + CE(goal_logits,goal)
  + 0.05 KL_RSSM.
```

Primary memory metric is `P(goal correct | paired current images identical)`. Ablation resets GRU hidden, resets RSSM deterministic+stochastic state, or limits Transformer history to one token. Pixel MSE is measured at horizons 1/5/10.

## Code Mapping

Dataset/alias condition: `benchmark_dataset.py`; matched adapters: `models.py`; objectives/training/timing/ablation: `run_benchmark.py`; raw evidence: `outputs/per_seed_results.csv`, `benchmark_results.json`.

## Training

Seeds 301/302/303; 128 train/64 test paired sequences; Adam `1e-3`; batch 32; 35 epochs/140 steps per model; identical weighted losses. Each checkpoint records model name, seed, config, optimizer, steps, and format version 1.

## Losses

Reconstruction anchors visual latent; next-image and latent consistency teach dynamics; Goal CE makes hidden information measurable; KL aligns RSSM posterior and prior. The memory Goal readout was explicitly standardized after an initial unfair version forced GRU/Transformer Goal information through the visually constrained latent.

## Evaluation Interface

`python 90_evaluation/01_memory_benchmark/run_benchmark.py` trains all 12 runs, records per-seed data, parameter bytes, batch-16 CPU latency, horizon metrics, and ablations.

## Results

| Model | Alias Goal accuracy mean±std | Ablated | h10 image MSE | Params | Batch-16 latency |
|---|---:|---:|---:|---:|---:|
| No Memory | 0.500±0.000 | 0.500 | 0.001147 | 336,994 | 1.95 ms |
| GRU | 0.833±0.236 | 0.500 | 0.001170 | 356,418 | 4.55 ms |
| RSSM | 1.000±0.000 | 0.500 | 0.001239 | 397,202 | 4.48 ms |
| Transformer | 1.000±0.000 | 0.500 | 0.001158 | 405,186 | 8.05 ms |

Parameter bytes are FP32 parameter storage lower bounds (1.35–1.62 MB), not peak process memory.

## Failure Cases

- GRU failed one seed completely, showing optimization instability.
- Pixel MSE barely separates methods and sometimes favors No Memory despite hidden-state failure.
- RSSM has worst mean h10 image MSE here despite perfect Goal memory.
- Transformer is ~1.8× recurrent latency at this short horizon.
- Three seeds estimate variability only coarsely.

## Findings

Memory is necessary for the intentionally aliased target; all memory ablations return to chance. More complex memory is not uniformly better: RSSM/Transformer are stable on Goal identity, but image error/latency tradeoffs differ. The model-state readout definition must be matched or the loss can suppress information memory actually holds.

## Limitations

Tiny two-Goal deterministic POMDP, auxiliary Goal supervision, CPU timing, no peak-memory profiler, and adapter implementations rather than unchanged paper code. RSSM uses posterior means and simplified Gaussian state; Transformer context is short.

## Compare Later

More seeds/budgets; no Goal supervision with probing; varying alias duration; noisy/stochastic environments; equal parameter budgets; free-nats/KL; context length; memory/activation profiling; success after planning.

## Final Model Candidate

```text
Candidate: RSSM for integration, Transformer as retained alternative.
Reason: Both were 3/3 stable on hidden Goal; RSSM was ~44% lower latency and supplies stochastic prior/posterior for uncertainty/imagination.
Advantages: RSSM combines stable memory and future prior; Transformer exposes history directly.
Disadvantages: RSSM image rollout was weakest; Transformer cost was highest; GRU was seed-unstable.
Possible conflicts: ensemble uncertainty multiplies RSSM compute; object slots require a structured RSSM state.
```

## Next Questions

Does RSSM remain preferred under matched parameter budgets and control success? Can ensemble uncertainty prevent imagination exploitation without excessive cost?

## References

PlaNet (Hafner et al., 2018, https://arxiv.org/abs/1811.04551), World Models (Ha & Schmidhuber, 2018, https://arxiv.org/abs/1803.10122), TransDreamer (Chen et al., 2022, https://arxiv.org/abs/2202.09481), and GRU (Cho et al., 2014, https://aclanthology.org/D14-1179/). Used only for model mechanisms; benchmark design/results are independent.
