# Deep RL Progress

Goal: train a neural target selector that beats both nearest-item BFS and the
current deterministic `rollout_gated` agent on average.

## Current Direction

Reset date: `2026-05-06`

The previous target-selection PPO became too complex: CNN board encoder,
behavior cloning, PPO-time teacher anchoring, and many route-inspired features.
That model produced useful results, but it was hard to explain and reason
about.

The active implementation is now a simpler PPO target selector inspired by the
plain architecture used in the PPO paper:

```text
reachable target features -> Linear(64) -> Tanh -> Linear(64) -> Tanh -> logit
```

The model chooses a target item. BFS then converts that target into the first
shortest-path movement action.

## Current Files

- `target_selection_ppo_mlp.py`
  - Model, target features, BFS target-to-action logic.
- `train_target_selection_ppo.py`
  - PPO rollout collection, updates, checkpointing, and evaluation.
- `record_target_selection_ppo.py`
  - Records replay JSON files from PPO checkpoints.
- `deterministic_agents/bfs_agent.py`
  - Nearest-item BFS baseline.
- `deterministic_agents/rollout_gated_agent.py`
  - Strong deterministic baseline and current leaderboard candidate.
- `recordings/`
  - Local replay outputs. Ignored by git except `.gitkeep`.
- `checkpoints/`
  - Local model checkpoints. Ignored by git except `.gitkeep`.
- `training_logs/`
  - Local long-run logs. Ignored by git except `.gitkeep`.

## Current Training Setup

Default opponent mix:

```text
bfs:0.5,rollout_gated:0.5
```

Default checkpoint:

```text
checkpoints/target_selection_ppo_mlp_latest.pt
```

Recommended training command from `final_project/finalproject/part2`:

```powershell
python .\train_target_selection_ppo.py --iterations 30 --rollout-steps 3000 --ppo-epochs 4 --batch-size 256 --learning-rate 2.5e-4 --entropy-coef 0.01 --opponent-mix bfs:0.5,rollout_gated:0.5 --eval-opponents bfs rollout_gated --eval-seeds 0 1 2 3 4 --checkpoint-path .\checkpoints\target_selection_ppo_mlp_latest.pt
```

Evaluate a checkpoint:

```powershell
python .\train_target_selection_ppo.py --eval-only --checkpoint-path .\checkpoints\target_selection_ppo_mlp_latest.pt --eval-opponents bfs rollout_gated --eval-seeds 0 1 2 3 4 5 6 7 8 9 --eval-steps 1000
```

Record a replay:

```powershell
python .\record_target_selection_ppo.py --checkpoint-path .\checkpoints\target_selection_ppo_mlp_latest.pt --opponent bfs --seed 0 --output .\recordings\target_selection_ppo_mlp_vs_bfs_seed0.json
```

## Success Criterion

A learned model is only worth packaging if it has positive mean score
difference against both:

- BFS over at least 10 fixed seeds
- rollout_gated over at least 10 fixed seeds

If that holds, test more seeds before replacing the deterministic submission
agent.

## Archived Result

The older CNN/behavior-cloning PPO checkpoint did reach positive average
performance over a short eval:

| Opponent | Seeds | Mean score diff | Min | Max |
| --- | --- | ---: | ---: | ---: |
| BFS | `0..9` | `+6.10` | `-9` | `+39` |
| rollout_gated | `0..9` | `+2.70` | `-8` | `+11` |

This result is kept as context only. It is not the active architecture anymore.

## Submission Note

Do not commit packaged agents or generated weights. Zip files, checkpoints,
recordings, logs, and generated reports are ignored by git.

If a PPO model becomes worth submitting, package locally as:

```text
agent.py
config.yaml
weights/model.pth
```
