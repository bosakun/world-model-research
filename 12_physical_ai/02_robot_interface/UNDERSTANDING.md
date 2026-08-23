# Understanding the Robot Boundary

## What problem does this solve?

It separates untrusted learned action requests from physical execution and defines exactly what experience is recorded for learning.

## Before / After / Core Idea

Before, model code called transitions directly. After, every action crosses deterministic safety checks and every transition has provenance/order. Safety is an external invariant, not something the policy is expected to learn.

## Data Flow

`observation -> demonstrator/policy -> requested action -> safety filter -> robot -> next observation -> replay`.

## Mathematics

Clipping limits magnitude; dead-man/workspace predicates replace action with zero. Inertial dynamics expose why action does not equal velocity instantly. Imitation MSE estimates demonstration behavior but does not optimize task reward.

## Code Mapping

`SafetyEnvelope.filter` is the actuator gate; `SimulatedMobileRobot.step` is replaceable hardware/simulator behavior; `DemonstrationDataset` records aligned transitions; `ImitationPolicy` is bounded twice; `replay_schema.json` is the data contract.

## Important Components

Dead-man enable, action bounds, workspace guard, immutable executed action in replay, source/episode/step metadata, simulator-first testing, and explicit hardware flag.

## What happens if we remove it?

- Envelope: policy bugs directly reach actuators.
- Executed-action logging: model trains on requested rather than physical control.
- Episode/step IDs: sequence boundaries corrupt memory training.
- Provenance: demonstrations and online actions cannot be audited.
- Simulator adapter: safety behavior is first tested on hardware.
- External approval: software success is incorrectly treated as execution authority.

## What I Should Be Able to Explain

- Why bound the policy and filter it again?
- Why record executed rather than requested action?
- What does dead-man stop protect?
- Why is transition-level validation weak?
- Why does 100% simulator success not authorize hardware?

## Questions

- How are timestamps/sensor delays represented?
- Which uncertainty threshold triggers emergency stop?
- How should recovery and operator takeover enter replay?
