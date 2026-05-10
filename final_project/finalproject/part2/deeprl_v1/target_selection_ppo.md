# Target-Selection PPO

Inspired by schulman et al (openai boys) from 2017

```text
observation -> reachable item targets -> MLP chooses target -> BFS returns move
```

The model learns target selection only. Movement toward the chosen target is
still handled by shortest-path BFS.

## Files

- `target_selection_ppo_mlp.py`
  - Model, target candidate features, BFS target-to-action logic, deterministic
    evaluation agent.
- `train_target_selection_ppo.py`
  - PPO rollout collection, updates, checkpointing, and evaluation.
- `record_target_selection_ppo.py`
  - Loads a checkpoint and records a replay.
- `deterministic_agents/bfs_agent.py`
  - Nearest-item BFS reference.
- `deterministic_agents/rollout_gated_agent.py`
  - Strong deterministic baseline and current submission candidate.

Training code is separate from model/inference code so the eventual upload
agent can reuse the small model without carrying the training loop.

## Dependencies

Only the project environment plus:

- NumPy
- PyTorch

No external RL libraries are used.

## Model Input

Each reachable item target gets a compact 27-dimensional feature vector:

- target position
- target position relative to self
- target position relative to opponent
- own BFS distance to target
- opponent BFS distance to target
- race margin
- whether this is the nearest item
- local cluster counts within BFS radius 3 and 5
- greedy route value after taking the target
- remaining item count
- whether the target is contested
- whether we are favored in the race
- whether the opponent is clearly favored in the race
- route value gated by own-favored race
- route value gated by lost race
- current score difference
- whether we are behind
- late-game behind pressure
- abandon pressure for losing races
- centrality of the target position
- edge proximity of the target position
- center-weighted cluster value
- edge-weighted cluster value

There is no board CNN and no learned pathfinding.

The critic also receives a 19-dimensional state-level feature vector. This is
not used by the actor. It exists because the value function was too weak when
it only saw pooled target rows. The global critic features include:

- own and opponent positions
- remaining item count
- min and mean own distance to targets
- min and mean opponent distance to targets
- fraction of contested targets
- fraction of targets we can reach before or at the same time as the opponent
- best route value in the current target set
- best local cluster counts
- current score difference
- whether we are behind
- current step progress
- remaining step fraction
- late-game behind pressure

## Neural Architecture

The actor is a shared target scorer:

```text
target_features -> Linear(64) -> Tanh -> Linear(64) -> Tanh -> Linear(1)
```

It is applied independently to each candidate target. The resulting logits form
a categorical policy over reachable item targets.

The critic uses the same style of two-hidden-layer tanh MLP, but its input is a
pooled summary of the candidate set plus the global state features:

```text
mean(target_features), max(target_features), min(target_features),
global_state_features, candidate_count
```

This keeps the architecture small while still letting the value function see
the whole target set.

## PPO Objective

The training script uses:

- clipped PPO policy objective
- value-function loss
- entropy bonus
- GAE advantages
- Adam optimizer

The default training reward is score-difference delta:

```text
new_score_diff - previous_score_diff
```

This focuses PPO on the competition objective: collecting more items than the
opponent. The raw environment reward difference is still available:

```powershell
--reward-mode env_diff
```

No behavior cloning or teacher anchor is active in this version.

### Shaped Score-Delta Reward

The trainer also supports:

```powershell
--reward-mode score_delta_shaped
```

This keeps score-difference delta as the base reward, then adds small dense
target-selection hints. The intent is to make cluster priority and
front-running less sparse without changing the final evaluation metric.

The shaping terms use the selected target's existing feature row:

```text
score_delta
+ own_favored_cluster_bonus
+ own_favored_route_bonus
+ front_run_collect_bonus
- lost_race_penalty
- lost_route_penalty
```

Default coefficients:

| Term | Default |
| --- | ---: |
| `--shaping-cluster-bonus` | `0.02` |
| `--shaping-route-bonus` | `0.03` |
| `--shaping-front-run-bonus` | `0.05` |
| `--shaping-lost-race-penalty` | `0.04` |
| `--shaping-lost-route-penalty` | `0.03` |
| `--shaping-max-abs` | `0.15` |

The shaping is clipped per step. The base score delta should still dominate;
the shaping is only there to tell PPO why a target choice was useful or bad.

## Training

From `final_project/finalproject/part2`:

```powershell
python .\train_target_selection_ppo.py --iterations 20 --rollout-steps 3000 --ppo-epochs 4 --batch-size 256 --learning-rate 2.5e-4 --entropy-coef 0.02 --reward-mode score_delta --opponent-mix bfs:0.4,rollout_gated:0.6 --eval-opponents bfs rollout_gated --eval-seeds 0 1 2 3 4 --checkpoint-path .\checkpoints\target_selection_ppo_mlp_abandon_latest.pt
```

The script saves:

- latest checkpoint: `--checkpoint-path`
- best checkpoint: same name with `_best.pt`

The best checkpoint is selected by the weakest mean score difference across
`--selection-opponents`. By default this is BFS and rollout_gated. This matters
when auxiliary training opponents are evaluated too, because they should not
necessarily define the submission metric.

## Position-Feature Migration From Best PPO

The current code can load older 23-feature checkpoints into the widened
27-feature model. The existing actor and critic input weights are copied into
the matching old feature columns. The new position-feature columns start with
zero input weight and are learned during the new run.

This makes it possible to restart from the strongest submitted PPO snapshot
without throwing away the useful abandonment-aware target-selection policy:

```powershell
python .\train_target_selection_ppo.py --resume-path .\checkpoints\target_selection_ppo_mlp_abandon_30_latest_best.pt --iterations 30 --rollout-steps 3000 --ppo-epochs 4 --batch-size 256 --learning-rate 1.0e-4 --entropy-coef 0.02 --reward-mode score_delta --opponent-mix bfs:0.25,rollout_gated:0.45,ppo_snapshot:0.30 --self-play-checkpoint .\checkpoints\target_selection_ppo_mlp_abandon_30_latest_best.pt --eval-opponents bfs rollout_gated ppo_snapshot --selection-opponents bfs rollout_gated --eval-seeds 0 1 2 3 4 --checkpoint-path .\checkpoints\target_selection_ppo_mlp_position_selfplay_latest.pt
```

This run deliberately uses `score_delta`, not `score_delta_shaped`, because the
previous shaped checkpoint looked strong on five seeds but did not generalize
better than the submitted PPO on ten seeds. The point of this experiment is to
test whether explicit target position information helps the old best policy
learn better cluster/route choices.

## Snapshot Self-Play

The trainer can include a frozen copy of a PPO checkpoint in the opponent pool:

```powershell
python .\train_target_selection_ppo.py --iterations 30 --rollout-steps 3000 --ppo-epochs 4 --batch-size 256 --learning-rate 2.5e-4 --entropy-coef 0.02 --reward-mode score_delta --opponent-mix bfs:0.25,rollout_gated:0.45,ppo_snapshot:0.30 --self-play-checkpoint .\checkpoints\target_selection_ppo_mlp_abandon_30_latest_best.pt --eval-opponents bfs rollout_gated ppo_snapshot --selection-opponents bfs rollout_gated --eval-seeds 0 1 2 3 4 --checkpoint-path .\checkpoints\target_selection_ppo_mlp_selfplay_snapshot_latest.pt
```

`ppo_snapshot` is frozen. The learner updates, but the snapshot opponent does
not. This is more stable than live self-play and keeps the experiment
interpretable. BFS and rollout_gated should remain in the mix so the model does
not overfit only to the snapshot policy.

Snapshot self-play with shaped reward:

```powershell
python .\train_target_selection_ppo.py --iterations 30 --rollout-steps 3000 --ppo-epochs 4 --batch-size 256 --learning-rate 2.5e-4 --entropy-coef 0.02 --reward-mode score_delta_shaped --opponent-mix bfs:0.25,rollout_gated:0.45,ppo_snapshot:0.30 --self-play-checkpoint .\checkpoints\target_selection_ppo_mlp_abandon_30_latest_best.pt --eval-opponents bfs rollout_gated ppo_snapshot --selection-opponents bfs rollout_gated --eval-seeds 0 1 2 3 4 --checkpoint-path .\checkpoints\target_selection_ppo_mlp_shaped_selfplay_latest.pt
```

## Evaluation

Evaluate without training:

```powershell
python .\train_target_selection_ppo.py --eval-only --checkpoint-path .\checkpoints\target_selection_ppo_mlp_abandon_latest_best.pt --eval-opponents bfs rollout_gated --eval-seeds 0 1 2 3 4 5 6 7 8 9 --eval-steps 1000
```

Record a replay:

```powershell
python .\record_target_selection_ppo.py --checkpoint-path .\checkpoints\target_selection_ppo_mlp_abandon_latest_best.pt --opponent bfs --seed 0 --output .\recordings\target_selection_ppo_mlp_abandon_vs_bfs_seed0.json
```

Visualize:

```powershell
python ..\src\visualiser\visualiser.py .\recordings\target_selection_ppo_mlp_abandon_vs_bfs_seed0.json
```

## Success Criterion

A learned model is worth packaging only if it has positive mean score
difference against both:

- BFS over at least 10 fixed seeds
- rollout_gated over at least 10 fixed seeds

If that holds, package:

```text
agent.py
config.yaml
weights/model.pth
```

All paths in `config.yaml` must be relative to the uploaded agent directory.
