# PPO v3 Handoff

This note summarizes the current target-selection PPO agent, feature set,
network inputs, training loop, opponent pool, and known results. It is intended
as a quick handoff for another agent to review the design and look for obvious
misses.

## Core Idea

The agent does not learn low-level grid movement. It learns which reachable item
to target next. For a selected item target, BFS converts the target into the
first shortest-path movement action.

Pipeline:

```text
observation
-> build reachable item candidates
-> actor scores each target candidate
-> sample target during training / argmax target during eval
-> BFS first action to selected target
```

Main files:

- `target_selection_ppo_mlp_v3.py`: feature construction, actor-critic model,
  inference wrapper, BFS helpers.
- `train_target_selection_ppo_v3.py`: rollout collection, rewards, GAE, PPO
  update, evaluation, checkpointing.
- `league_train_v3.py`: repeated league generations, gate evaluation, accepted
  checkpoint pool.
- `league_pool.json`: accepted frozen league snapshots.

## Feature-Building Functions

| Function | Purpose |
| --- | --- |
| `get_state_parts` | Extracts tile map, own position, opponent position, and item positions from the observation. |
| `get_score_context` | Builds score/time context: normalized score diff, behind flag, normalized step, time remaining, urgent-behind flag. |
| `valid_neighbors` | Enumerates legal non-obstacle neighboring cells. |
| `safe_random_action` | Fallback action if there are no candidate targets. |
| `bfs_distances_and_first_actions` | Computes BFS distances from a start cell and the first action on the shortest path to every reachable cell. |
| `normalized_distance` | Clips distances by `MAX_DISTANCE = 32` and maps them to `[0, 1]`; unreachable is `1.0`. |
| `count_cluster` | Counts nearby item targets around a candidate within a BFS radius. |
| `nearby_cluster_items` | Returns items within a BFS-radius neighborhood around a target. |
| `route_value` | Greedy lookahead feature for whether a target opens a short follow-up route; depth `3`, decay `0.7`. |
| `position_features` | Computes center/edge scores and cluster-position interactions. |
| `cluster_control_features` | v3 cluster-control features: controlled cluster count, cluster race margin, singleton-tunnel flag, cluster swing value. |
| `build_global_features` | Builds the 19-dimensional state-level critic feature vector. |
| `build_target_candidates` | Main feature assembly function; returns `TargetCandidates(features, global_features, targets, first_actions)`. |

## Per-Target Actor Features

`FEATURE_DIM = 31`. Built in `build_target_candidates`. One row is created for
each reachable item target.

| Index | Name | Meaning |
| ---: | --- | --- |
| 0 | `target_y_norm` | Target row normalized to `[-1, 1]`. |
| 1 | `target_x_norm` | Target column normalized to `[-1, 1]`. |
| 2 | `target_dy_self` | Target row relative to own position, divided by `15`. |
| 3 | `target_dx_self` | Target column relative to own position, divided by `15`. |
| 4 | `target_dy_opp` | Target row relative to opponent position, divided by `15`. |
| 5 | `target_dx_opp` | Target column relative to opponent position, divided by `15`. |
| 6 | `own_distance` | Own BFS distance to target, normalized by `32`. |
| 7 | `opponent_distance` | Opponent BFS distance to target, normalized by `32`; unreachable becomes `1.0`. |
| 8 | `race_margin` | `(opponent_distance - own_distance) / 32`, clipped to `[-1, 1]`; positive means own-favored. |
| 9 | `nearest_flag` | `1` if this is one of the nearest reachable items for us. |
| 10 | `cluster3` | Count of item targets within BFS radius 3, clipped by 10 and normalized. |
| 11 | `cluster5` | Count of item targets within BFS radius 5, clipped by 16 and normalized. |
| 12 | `route_value` | Greedy short-route value after collecting this target. |
| 13 | `item_count_norm` | Remaining reachable item count, clipped at 32 and normalized. |
| 14 | `contested` | `1` if `abs(race_margin_raw) <= 2`. |
| 15 | `own_favored` | `1` if `opponent_distance - own_distance >= 0`. |
| 16 | `lost_race` | `1` if opponent is favored by more than one step. |
| 17 | `route_if_own_favored` | `route_value * own_favored`. |
| 18 | `route_if_lost` | `route_value * lost_race`. |
| 19 | `score_diff_norm` | `(own_score - opponent_score) / 50`, clipped to `[-1, 1]`. |
| 20 | `behind` | `1` if own score is lower. |
| 21 | `urgent_behind` | `behind * steps_norm`. |
| 22 | `abandon_pressure` | `lost_race * (0.5 + 0.5 * behind)`. |
| 23 | `center_score` | Higher for targets closer to map center. |
| 24 | `edge_score` | Higher for targets closer to map edge. |
| 25 | `center_cluster_value` | `center_score * cluster_value`. |
| 26 | `edge_cluster_value` | `edge_score * cluster_value`. |
| 27 | `cluster_control_value` | Normalized count of nearby radius-5 cluster items our agent reaches no later than opponent. |
| 28 | `cluster_race_margin` | Mean BFS race margin over the target's radius-5 cluster, normalized by `32`. |
| 29 | `singleton_tunnel_flag` | `1` for contested low-cluster singleton targets with nearly tied immediate race. |
| 30 | `cluster_swing_value` | `cluster_control_value - 0.5 * singleton_tunnel_flag`, clipped to `[-1, 1]`. |

Important derived values:

- `cluster_value = 0.5 * (cluster3_norm + cluster5_norm)`.
- `race_margin_raw = opponent_distance - own_distance`.
- `cluster_control_features` computes `cluster_distance_margin` internally, but
  it is currently not returned or used.

## Critic Global Features

`GLOBAL_FEATURE_DIM = 19`. Built in `build_global_features`.

| Index | Name | Meaning |
| ---: | --- | --- |
| 0 | `self_y_norm` | Own row normalized to `[-1, 1]`. |
| 1 | `self_x_norm` | Own column normalized to `[-1, 1]`. |
| 2 | `opponent_y_norm` | Opponent row normalized to `[-1, 1]`. |
| 3 | `opponent_x_norm` | Opponent column normalized to `[-1, 1]`. |
| 4 | `item_count_norm` | Reachable item count, clipped at 32 and normalized. |
| 5 | `own_min_distance` | Minimum own target distance, normalized. |
| 6 | `own_mean_distance` | Mean own target distance, normalized. |
| 7 | `opponent_min_distance` | Minimum opponent target distance, normalized. |
| 8 | `opponent_mean_distance` | Mean opponent target distance, normalized. |
| 9 | `contested_fraction` | Fraction of candidates with absolute raw race margin <= 2. |
| 10 | `own_favored_fraction` | Fraction of candidates where race margin is non-negative. |
| 11 | `max_route_value` | Best route value among candidates. |
| 12 | `max_cluster3` | Best radius-3 cluster value, clipped/normalized. |
| 13 | `max_cluster5` | Best radius-5 cluster value, clipped/normalized. |
| 14 | `score_diff_norm` | Normalized current score difference. |
| 15 | `behind` | Behind flag. |
| 16 | `steps_norm` | Current step divided by 1000, clipped. |
| 17 | `steps_remaining_norm` | `1 - steps_norm`. |
| 18 | `urgent_behind` | `behind * steps_norm`. |

## Network Architecture

Class: `TargetSelectionMLP`.

Actor:

```text
target_features(31)
-> Linear(31, 64)
-> Tanh
-> Linear(64, 64)
-> Tanh
-> Linear(64, 1)
```

The same actor MLP is applied to every valid item candidate. The output is one
logit per candidate. Invalid padded candidates are masked with `-1e9`.

During training:

```python
dist = Categorical(logits=logits)
action_index = dist.sample()
```

During evaluation:

```python
action_index = torch.argmax(logits, dim=1)
```

Critic:

```text
mean(target_features)  31
max(target_features)   31
min(target_features)   31
global_features        19
candidate_count         1
-------------------------
critic input           113
-> Linear(113, 64)
-> Tanh
-> Linear(64, 64)
-> Tanh
-> Linear(64, 1)
```

The critic outputs one scalar `V(s)` for the whole candidate state. It is not
fed the actor logits.

## Training Loop

Main functions:

- `collect_rollout`: samples opponents from the current opponent mix, runs the
  environment, samples target choices, stores transitions.
- `training_reward`: base reward is either environment reward diff or score
  delta. Current mode is `score_delta_cluster`.
- `cluster_signal_reward`: current dense shaping for cluster control.
- `compute_gae`: computes advantages and value targets with GAE.
- `ppo_update`: clipped PPO objective, value MSE, entropy bonus.
- `evaluate_many`: deterministic evaluation by argmax target.
- `selection_score_from_metrics`: saves best checkpoints based on mean/win gates.

Current active hyperparameters from `league_gen07_latest.pt`:

| Parameter | Value |
| --- | ---: |
| `iterations` | `20` |
| `rollout_steps` | `5000` |
| `ppo_epochs` | `4` |
| `batch_size` | `256` |
| `hidden_dim` | `64` |
| `learning_rate` | `2.5e-4` |
| `gamma` | `0.99` |
| `gae_lambda` | `0.95` |
| `clip_coef` | `0.2` |
| `value_coef` | `0.5` |
| `entropy_coef` | `0.02` |
| `max_grad_norm` | `0.5` |
| `reward_mode` | `score_delta_cluster` |
| `cluster_signal_max_abs` | `0.50` |
| `terminal_win_bonus` | `5.0` |
| `eval_seeds` | `0..4` |

Current `score_delta_cluster` shaping terms:

```text
+ center-cluster bonus
+ own-favored route bonus
+ cluster-control bonus
+ positive cluster-swing bonus
- singleton-tunnel penalty
- lost-cluster penalty
```

Default coefficients in `train_target_selection_ppo_v3.py`:

| Term | Default |
| --- | ---: |
| `cluster_signal_center_bonus` | `0.08` |
| `cluster_signal_route_bonus` | `0.06` |
| `cluster_signal_control_bonus` | `0.12` |
| `cluster_signal_swing_bonus` | `0.08` |
| `cluster_signal_lost_penalty` | `0.08` |
| `singleton_tunnel_penalty` | `0.08` |

## Opponent Pool

Opponent builder: `build_opponent`.

Supported opponents:

- `bfs`: deterministic nearest-reachable-item BFS agent.
- `rollout_gated`: stronger deterministic target-selection agent with cluster,
  route, and opponent-position heuristics.
- `ppo`: older frozen PPO checkpoint,
  `part2/checkpoints/target_selection_ppo_mlp_abandon_30_latest_best.pt`.
- `position_aware_ppo`: older frozen position-aware PPO checkpoint,
  `part2/checkpoints/target_selection_ppo_mlp_position_selfplay_latest_best.pt`.
- `league`: accepted frozen snapshots from `league_pool.json`.
- `baseline`, `random`: provided agents, mostly for eval.

Current gen07 training mix:

```text
bfs                 0.09
rollout_gated       0.15
ppo                 0.15
position_aware_ppo  0.21
league              0.40
```

The base mix is scaled down by `1 - league_weight`. With `league_weight=0.40`,
the accepted league bucket is relatively strong.

Accepted league snapshots in `league_pool.json`:

| Generation | Checkpoint | Gate means over 10 seeds |
| ---: | --- | --- |
| 1 | `checkpoints/ppo_league_gen1.pt` | BFS `+17.6`, rollout_gated `+6.2`, PPO `+6.6`, position-aware PPO `+6.0` |
| 5 | `checkpoints/ppo_league_gen5.pt` | BFS `+16.0`, rollout_gated `+11.0`, PPO `+11.4`, position-aware PPO `+2.5` |

With recency decay `0.75` and two accepted snapshots, the internal league bucket
should sample gen5 more often than gen1.

## Results Snapshot

### Packaged submission candidate

Checkpoint: `checkpoints/ppo_league_gen2_iter14.pt`.

Five-seed training eval stored in checkpoint:

| Opponent | Mean score diff | Wins |
| --- | ---: | ---: |
| BFS | `+8.8` | `5/5` |
| rollout_gated | `+10.6` | `4/5` |
| PPO | `+12.2` | `5/5` |
| position-aware PPO | `+8.4` | `4/5` |
| league | `+3.0` | `4/5` |

Ten-seed validation in `training_logs/league_gen02_eval10.log`:

| Opponent | Mean score diff | Min | Max |
| --- | ---: | ---: | ---: |
| BFS | `+7.5` | `-13` | `+18` |
| rollout_gated | `+9.6` | `-12` | `+32` |
| PPO | `+7.0` | `-16` | `+24` |
| position-aware PPO | `+3.2` | `-10` | `+19` |

Manual baseline/random eval over seeds `0..4`:

| Opponent | Mean score diff | Min | Max | Wins |
| --- | ---: | ---: | ---: | ---: |
| baseline | `+85.2` | `+62` | `+98` | `5/5` |
| random | `+143.8` | `+130` | `+152` | `5/5` |

### Best accepted league checkpoint

Checkpoint: `checkpoints/ppo_league_gen5.pt`, accepted from
`league_gen05_latest_best.pt`.

Ten-seed gate eval in `training_logs/league_gen05_eval10.log`:

| Opponent | Mean score diff | Wins |
| --- | ---: | ---: |
| BFS | `+16.0` | `10/10` |
| rollout_gated | `+11.0` | `9/10` |
| PPO | `+11.4` | `9/10` |
| position-aware PPO | `+2.5` | `7/10` |

Five-seed checkpoint metrics:

| Opponent | Mean score diff | Wins |
| --- | ---: | ---: |
| BFS | `+23.0` | `5/5` |
| rollout_gated | `+10.2` | `4/5` |
| PPO | `+17.2` | `5/5` |
| position-aware PPO | `+6.4` | `5/5` |
| league | `+6.4` | `4/5` |

### Current active gen07 run

Current latest checkpoint: `checkpoints/league_gen07_latest.pt`, iteration `14`.
Current best checkpoint: `checkpoints/league_gen07_latest_best.pt`, iteration `1`.

Best gen07 checkpoint metrics:

| Opponent | Mean score diff | Wins |
| --- | ---: | ---: |
| BFS | `+21.0` | `5/5` |
| rollout_gated | `+9.2` | `4/5` |
| PPO | `+14.2` | `5/5` |
| position-aware PPO | `+5.0` | `4/5` |
| league | `+4.8` | `4/5` |

Latest gen07 checkpoint metrics at iteration 14:

| Opponent | Mean score diff | Wins |
| --- | ---: | ---: |
| BFS | `+8.0` | `3/5` |
| rollout_gated | `+5.0` | `3/5` |
| PPO | `-0.2` | `3/5` |
| position-aware PPO | `+3.0` | `3/5` |
| league | `+2.4` | `4/5` |

Interpretation: gen07 is currently volatile. The saved best is still iteration
1 because later checkpoints did not beat its selection score.

## Checkpoint Selection and Gates

Training-time best checkpoint selection for current league run:

```text
selection_opponents = position_aware_ppo, league
selection_mean_opponents = bfs, rollout_gated, ppo, position_aware_ppo, league
selection_min_wins = 3 over 5 eval seeds
selection_min_mean_score = 0.0
```

League accept/reject gate from `league_train_v3.py` defaults/current docs:

```text
gate seeds = 0..9
mean gate opponents = bfs, rollout_gated, ppo, position_aware_ppo
win gate opponents = bfs, rollout_gated, ppo, position_aware_ppo
mean_gate_min_score = 0.0
win_gate_min_wins = 6/10
```

Older docs mention a stricter default target of improving beyond previous
position-aware PPO target results. The active pool currently records
`best_score = 2.5` in `league_pool.json`, based on the accepted gen5 minimum
mean gate score.

## Review Targets / Possible Obvious Misses

- `compute_gae` appends `0.0` as the final bootstrap value. If a rollout ends
  mid-episode, the last transition is treated as if the next value is zero
  rather than bootstrapping from the current critic. This may add bias at rollout
  boundaries.
- Training-time eval repeatedly uses seeds `0..4`; gate eval uses `0..9`.
  Strong checkpoints may be partially selected for those seeds. More holdout
  seeds would give a cleaner signal.
- The policy learns target selection only. It cannot learn tactical low-level
  movement such as blocking, waiting, or intentionally taking a non-shortest
  route unless that behavior is expressible through target choice plus BFS.
- `cluster_distance_margin` is computed inside `cluster_control_features` but is
  not returned or used.
- The active gen07 latest checkpoint has regressed relative to its saved best.
  Use `league_gen07_latest_best.pt` for evaluation, not `league_gen07_latest.pt`,
  unless there is a specific reason to inspect the latest weights.
- The critic sees pooled mean/max/min target features. This is fixed-size and
  simple, but it loses candidate-set structure and pairwise target relationships.
- Baseline/random results were only checked over five seeds in the manual eval.
  That is enough for report evidence, but not enough to estimate variance.
