# Generated outputs

- `full_world.png`: evaluator-visible 5x5 true world.
- `partial_observation.png`: corresponding Agent-visible 3x3 local image; blue means unknown.
- `observation_sequence.png`: `t=0..2` full/partial sequence demonstrating Goal disappearance.
- `aliasing_pair.png`: two histories with different true Goals and bitwise-identical partial observation at `t=2`.

Regenerate with:

```bash
uv run python 03_memory/02_partial_observation/visualize.py
```

