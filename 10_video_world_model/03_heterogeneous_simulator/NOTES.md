# Research Notes

## Before

- Expected a typed adapter to be more reliable than concatenating motor/language/goal with zero placeholders.
- Needed to distinguish “heterogeneous schemas” from Phase 11 simultaneous multimodal fusion.

## Results

- Four tests passed.
- Unweighted MSE motor/language/goal: `0.01400 / 0.01417 / 0.01427`.
- Weighted validation losses also remained close (`~0.037`).
- Comparable performance is expected because all sources ultimately describe the same five motions; it is a plumbing/mechanism result.

## Implementation insights

- Adapter isolation test changes motor values by +100 and verifies language/goal selected conditions are bitwise unaffected.
- Per-source curves are more informative than one averaged training curve.

## Article material

- `heterogeneous_predictions.png`: three schemas, one prediction interface.
- Explain why provenance/type masks are semantic data, not batching boilerplate.

## Compare later

Source imbalance, missing/conflicting inputs, domain-shifted renderers, adapter ablations, shared versus separate models.
