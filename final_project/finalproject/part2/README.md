# Part 2 Agent Experiments

This folder is structured so deterministic baselines and deep-RL experiments
can be worked on separately.

## Layout

- `deterministic_agents/`
  - Non-deep agents and deterministic baselines.
  - Includes BFS and rollout-gated agents.
- `target_selection_ppo_mlp.py`
  - Small target-selection PPO model and inference helpers.
- `train_target_selection_ppo.py`
  - Training loop for the MLP PPO target selector.
- `record_target_selection_ppo.py`
  - Records a replay from a PPO checkpoint.
- `record_rollout_gated.py`
  - Records a replay from the deterministic rollout-gated agent.
- `evaluate_rollout_gated.py`, `sweep_rollout_gated.py`
  - Deterministic-agent evaluation and hyperparameter search utilities.
- `recordings/`
  - Generated replay JSON files.
- `checkpoints/`
  - Generated model checkpoints.
- `training_logs/`
  - stdout/stderr logs from longer training runs.
- `autoresearch_reports/`
  - Generated rollout-gated search/evaluation reports.

## Git Policy

Source code and documentation should be committed. Generated checkpoints,
logs, recordings, and reports are ignored by default because they can become
large and machine-specific.

The directories stay in git through `.gitkeep` files.
