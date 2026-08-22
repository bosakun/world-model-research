# Understanding Temporal Abstraction

## What problem does this solve?

Long prediction can require too many recursive primitive transitions. Temporal abstraction models change over several primitive steps in one learned transition.

## Before

Thirty primitive actions require thirty model calls, each consuming a prediction from the previous call.

## After

Five ordered actions become one macro condition. Thirty primitive steps require six macro calls, evaluated at state boundaries `0,5,...,30`.

## Core Idea

Temporal abstraction trades resolution for depth. Fewer predictions can reduce composition count, but each macro prediction is harder and hides within-chunk events.

## Data Flow

```text
[a_t,...,a_{t+4}] -> GRU -> c_t
s_t + c_t -> MacroDynamics -> s_hat_{t+5}
```

## Mathematics

```text
s_{t+5}=F(F(F(F(F(s_t,a_t),a_{t+1}),a_{t+2}),a_{t+3}),a_{t+4})
c_t=Encoder(a_t,...,a_{t+4})
s_hat_{t+5}=s_t+g_theta(s_t,c_t).
```

- `F`: primitive true transition.
- `c_t`: learned ordered chunk representation.
- `g_theta`: macro residual model.

Why needed: approximating the composition directly reduces inference depth.

Teacher-forced macro loss is

```text
L_macro=E ||s_hat_{t+5}-s_{t+5}||^2.
```

It does not supervise intermediate `s_{t+1...t+4}`.

## Code Mapping

| Concept | Code |
|---|---|
| primitive boundaries/action chunks | `macro_dataset.py::chunk_sequences` |
| ordered chunk embedding | `macro_dynamics.py::ActionChunkEncoder` |
| macro residual | `MacroDynamics.forward` |
| recursive macro prediction | `MacroDynamics.rollout` |
| boundary-only evaluation | `evaluate.py` |

## Important Components

- Ordered encoder: action permutations can change nonlinear outcomes.
- Boundary contract: makes exactly which states exist at macro resolution explicit.
- Residual prediction: anchors macro output to input state.
- Primitive-equivalent horizon labels: six macro steps must be reported as 30 primitive steps, not “horizon 6” without context.

## What happens if we remove it?

- Remove action order: different trajectories with the same action counts alias.
- Use only one action: the macro predictor lacks four controls.
- Predict intermediate states too: no longer a boundary-only abstraction and requires another output contract.
- Increase chunk size blindly: fewer calls but harder approximation and more hidden events.
- Treat fixed chunks as full options: incorrectly claims policies/termination that do not exist.

## Options versus this experiment

An option normally includes an initiation set, internal policy, and termination condition. This experiment receives a predetermined open-loop action chunk of fixed length five. It studies a macro **world-model transition**, not option discovery or hierarchical control.

## What I Should Be Able to Explain

- What information is inside one macro token?
- Why must action order be preserved?
- Why are there seven boundary states but six chunks?
- What does the model know about intermediate time steps? Nothing explicit.
- How can fewer model calls still yield large error?
- Why is comparing macro horizon 6 to primitive horizon 6 misleading?
- Which mechanisms from the Options framework are absent?
- Why will reward and termination complicate macro planning?

## Questions

- Should temporal duration be learned or selected by a hierarchy?
- How can intermediate safety constraints be checked?
- Should macro outputs include cumulative reward and continuation probability?
- Can primitive and macro predictions be consistency-regularized?
- When does chunk encoding need attention instead of a GRU?
