# Understanding the Unified Memory Comparison

## What problem does this solve?

It distinguishes “predicts common pixels” from “retains hidden world state,” while keeping dataset/training/evaluation differences from masquerading as architecture effects.

## Before / After / Core Idea

Before, every model had its own smoke test. After, all models receive the same aliases and losses. Evaluate a variable that identical current images cannot reveal, then erase memory and require the advantage to disappear.

## Data Flow

`paired histories -> equal current observation/different Goal -> two observed context transitions -> ten imagined steps -> pixel and hidden-Goal metrics -> memory ablation`.

## Mathematics

Conditional alias accuracy measures `P(y_hat=y | o_t(pair0)=o_t(pair1))`; a current-frame function cannot systematically exceed 0.5 on balanced contradictory labels. Memory can because histories differ before aliasing.

Mean/std across seeds exposes stability. Horizon MSE measures compounding visual error. Parameter bytes=`4*parameter_count` is only model-weight storage. Latency measures repeated batch-16 rollout on this machine.

## Code Mapping

`alias_mask` proves equality from tensors; each `rollout` consumes exactly the same context/actions; `ablate=True` removes the architecture's information carrier; aggregation never averages different model tasks.

## Important Components

Paired aliases create an identifiable need for memory. Shared Goal readouts must access equivalent memory states. Multiple seeds prevent one lucky initialization. Ablation establishes mechanism dependence. Raw per-seed CSV prevents mean-only storytelling.

## What happens if we remove it?

- Alias conditioning: pixel accuracy can hide memory failure.
- Multiple seeds: GRU would appear either perfect or useless depending on seed.
- Memory ablation: correlation cannot establish use.
- Matched readout: visually constrained latent unfairly suppresses GRU/Transformer memory.
- Horizon metrics: one-step quality hides compounding error.
- Raw results: aggregate claims cannot be audited.

## What I Should Be Able to Explain

- Why must No Memory be at chance on balanced aliases?
- Why did the initial Goal-head placement make comparison unfair?
- Why does similar image MSE not imply similar world state?
- What does the ablation establish?
- Why is GRU mean 0.833 with large std?
- Why are parameter bytes not peak memory usage?

## Questions

- Would a frozen probe replace Goal-supervised training more cleanly?
- How many seeds are needed for GRU failure probability?
- Does memory quality improve downstream planning success?
