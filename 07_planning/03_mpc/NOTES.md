# MPC Notes

- CEM-MPC reached the deterministic Goal in 12 feedback steps.
- Each action caused a complete replan; replanning calls equal executed actions.
- `outputs/mpc_rollout.png` is the first explicit closed loop back to an environment in this roadmap.
- Main future experiment: inject transition bias/noise and compare open-loop versus MPC correction.
- Physical AI will require latency measurement and a fallback controller.
