# Understanding C-SWM

## What problem does this solve?

It makes entities explicit so the model can reuse one transition rule per object and one relation rule per object pair.

## Before

One latent vector represented the entire scene. “Which object changed?” and “which pair interacted?” had no explicit tensor dimension.

## After

State has shape `[objects,features]`. Each object receives its action and aggregated messages from the others, yielding an object-wise predicted future.

## Core Idea

Structure is an inductive bias: factor the scene by entity, share functions across entities, and learn relations with messages. Contrastive energy trains predictiveness without decoding every pixel.

## Data Flow

`image -> known object binding -> slots -> graph transition(action) -> predicted slots -> positive/negative energies`.

## Mathematics

`m_i=sum_(j!=i)g(z_i,z_j)` aggregates all senders. It is needed because object `i` may change because of object `j`.

`z_hat_i'=z_i+f(z_i,a_i,m_i)` combines persistence, direct action, and relation. Shared `f` applies the same physical rule to every object.

`L=E+ + max(0,gamma+E+-E-)` makes the true future close and an unrelated scene farther than margin `gamma`. Without the negative term, all observations can encode to the same constant.

## Code Mapping

`ColorObjectEncoder` creates the object axis; `RelationalTransition.edge` computes pair effects; `node` updates slots; `contrastive_world_model_loss` implements energy and margin.

## Important Components

Slots separate entities; weight sharing supports compositional reuse; edges model interaction; residual updates express persistence; negatives protect against collapse. Known color binding isolates dynamics but is not a general perception solution.

## What happens if we remove it?

- Object axis: return to entangled global state.
- Edge network: collision/repulsion must be inferred independently inside each node.
- Shared functions: object index becomes a separate task.
- Action input: predicted movement averages across controls.
- Negative energy: constant slots can minimize positive transition energy.
- Fixed color binding: slot discovery becomes necessary and ordering may permute.

## What I Should Be Able to Explain

- Why is a set of slots different from splitting one vector into chunks?
- What information travels in an edge message?
- Why can positive-only energy collapse?
- Why does a linear probe not train the representation?
- Which part of object perception is deliberately skipped here?

## Questions

- How do we match predicted and observed slots under permutations?
- Which negatives are genuinely hard rather than merely different positions?
- Can relations transfer from two to three objects?
