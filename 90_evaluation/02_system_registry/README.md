# Cross-Phase Evaluation Registry

Status: completed on 2026-08-23. Evidence catalog, not a leaderboard.

## Purpose

Make every generated evaluation artifact discoverable with dataset version, seed, entry point, and raw metrics while preventing incomparable tasks from being ranked together.

## Problem / Previous Model / Hypothesis

Results were distributed across experiment folders. A registry should expose coverage and missing evidence without flattening uncertainty calibration, planning return, segmentation IoU, and robot success into one false scalar.

## Architecture / Data Flow

```text
repository/**/outputs/evaluation_metrics.json
 -> validate metadata -> preserve complete payload
 -> experiment_registry.json + flat CSV + phase coverage plot
```

## Tensor Shapes / Mathematics

No model tensors. Registry rows are experiments. Numeric metrics remain keyed dictionaries. Phase counts are `count(executable evaluation artifacts)` and are not quality scores.

## Code Mapping

Discovery/metadata extraction: `build_registry.py::discover`; serialization/plot: `build`. Source fixes added dataset/entry metadata to legacy GRU and three planning outputs.

## Training / Losses

None. This is read-only aggregation over local evidence.

## Evaluation Interface

`python 90_evaluation/02_system_registry/build_registry.py` currently catalogs 24 experiments across Phases 03–12 and 99. Phase 90's matched benchmark is stored separately because its artifact aggregates 12 runs rather than one experiment metric file.

## Smoke Test Results

Two tests passed: required cross-phase artifacts are found, and every registered row now records dataset version and evaluation entry point.

## Results / Findings

Coverage: Memory 3, Uncertainty 2, Long Horizon 2, Reward/Value 1, Planning 4, Imagination 1, Spatial 4, Video 3, Multimodal 1, Physical AI 2, Integrated 1. The registry makes one-seed evidence visibly distinct from the matched three-seed memory benchmark.

## Failure Cases / Limitations

Artifact presence is not correctness. Metric names/units are heterogeneous. Checkpoints are ignored by Git and must be regenerated. Parameter bytes are not peak memory. Registry excludes training-only outputs without evaluation JSON.

## Compare Later

Use task-specific protocols. Do not compare voxel IoU numerically against planning success or reward RMSE.

## Final Model Candidate

```text
Candidate: Yes as research infrastructure.
Reason: Traceability is required for evidence-based integration.
Advantages: auditable provenance; machine-readable coverage.
Disadvantages: cannot standardize fundamentally different tasks.
Possible conflicts: schema evolution requires registry versioning.
```

## Next Questions

How should future OOD and physical evaluations upgrade Phase 99's one-seed evidence?

## References

No algorithm is implemented. Metrics retain references in their originating experiment README.
