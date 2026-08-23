# Understanding the Evaluation Registry

## What problem does this solve?

It separates evidence discovery from scientific comparison.

## Before / After / Core Idea

Before, results required manual folder search. After, one manifest points to raw artifacts and metadata. Preserve metric meaning instead of inventing a universal score.

## Data Flow / Mathematics / Code Mapping

`rglob -> JSON parse -> metadata row -> lossless nested JSON + compact CSV`. Counts measure coverage only. `discover` excludes its own generated outputs to avoid recursion.

## Important Components

Dataset version, seed, evaluation entry point, relative path, raw payload, registry version, and explicit limitations.

## What happens if we remove it?

Missing metadata goes unnoticed; one-seed results look equivalent to multi-seed evidence; integration decisions become hard to audit. If heterogeneous metrics are normalized into one score, their semantics are destroyed.

## What I Should Be Able to Explain

- Why is this not a leaderboard?
- What evidence does artifact presence provide?
- Why are raw nested metrics retained?
- Why is Phase 90 memory evidence separate?

## Questions

- Should future schemas validate metric units/types?
- How should regenerated checkpoint hashes be tracked?
