# Understanding Heterogeneous Simulator Data

## What problem does this solve?

It gives incompatible condition schemas a typed route into one shared dynamics interface.

## Before

Each experiment assumed one uniform action representation. Combining datasets would make “unused zero” indistinguishable from a real zero action.

## After

A source/type variable selects exactly one adapter. The shared simulator always receives the same condition shape, while evaluation remains stratified by source.

## Core Idea

Normalize interfaces, not raw meanings: modality-specific adapters translate each schema; a shared transition learns common consequences; source metadata preserves provenance.

## Data Flow

`source record -> selected adapter + source embedding -> shared condition -> image latent transition -> next image`.

## Mathematics

`c=A_type(input)+e_type` uses a discrete selector, avoiding concatenated dummy fields. `z'=tanh(z+f(z,c))` is shared across sources. Source-specific error `E[L|type=k]` prevents a majority source from hiding failure.

## Code Mapping

`kind` is provenance; `motor/language/goal` are schemas; `condition` stacks then selects adapters; evaluation filters each source independently.

## Important Components

Typed selection prevents unused-field leakage. Shared latent/dynamics enables transfer. Source embedding allows systematic schema offsets. Balanced sampling and per-source metrics expose negative transfer.

## What happens if we remove it?

- Type selector: zeros/defaults become ambiguous.
- Adapters: incompatible shapes/semantics are forced together.
- Shared dynamics: no cross-source reuse.
- Source metrics: one easy/large dataset can hide failure.
- Provenance embedding: shared semantics are assumed perfectly aligned.

## What I Should Be Able to Explain

- Why is missing input not the same as numerical zero?
- What does an adapter normalize?
- Why can source embeddings both help and create shortcuts?
- Why report worst-source error?
- How is selection different from simultaneous multimodal fusion?

## Questions

- Can shared semantics emerge when domains/images differ?
- How should conflicting conditions be reconciled?
- Which source balancing strategy scales to real corpora?
