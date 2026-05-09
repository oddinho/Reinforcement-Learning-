# Deep RL v2

This folder is an isolated second-generation PPO experiment. It intentionally
does not modify the existing part2 PPO files.

## Goal

Train a deeper target-selection PPO agent that can beat:

- nearest-item BFS
- deterministic `rollout_gated`
- the previous submitted PPO
- the current position-aware PPO

The position-aware PPO is included in the opponent pool because its 10-seed
eval improved the weaker opponent score:

| Agent | BFS `0..9` | rollout_gated `0..9` |
| --- | ---: | ---: |
| previous submitted PPO | `+9.90` | `+4.70` |
| position-aware PPO | `+9.00` | `+6.70` |

## Architecture

The model keeps the target-selection setup:

```text
observation -> reachable item targets -> MLP chooses target -> BFS moves
```

The policy still does not learn pathfinding. BFS converts the chosen target
into the first shortest-path movement action.

The actor is now a deeper-capacity version of the previous MLP:

```text
target_features -> Linear(128) -> Tanh -> Linear(128) -> Tanh -> Linear(1)
```

The critic uses the same hidden size and receives:

```text
mean(target_features),
max(target_features),
min(target_features),
global_state_features,
candidate_count
```

The target feature set is the 27-dimensional position-aware feature set from
the previous PPO:

- race and distance features
- local cluster features
- route value
- score/behind pressure
- abandon-pressure features
- center/edge position features

## Reward

The base reward remains score-difference delta:

```text
new_score_diff - previous_score_diff
```

v2 adds an episode-level win incentive. On the terminal transition only:

```text
if final_score_diff > 0:
    reward += 5.0
```

There is no terminal loss penalty in this first v2 run. The intent is to align
training better with the competition objective: winning games matters more
than only maximizing score margin.

The log prints this as:

```text
mean_terminal_bonus=...
```

## v2.1 Cluster Signal

Replay inspection of the best v2 checkpoint still showed a recurring weakness:
the policy often preferred nearby single targets over route-opening clusters.
This is plausible because cluster payoff is delayed, while the base
score-delta reward immediately reinforces single pickups.

Implemented a focused reward mode:

```text
score_delta_cluster
```

This keeps score-difference delta as the base reward, keeps the terminal win
bonus, and adds a clipped target-choice signal:

```text
cluster_signal =
    0.08 * own_favored * center_cluster_value
  + 0.06 * own_favored * route_value
  - 0.08 * lost_race * cluster_value
```

Default coefficients:

| Term | Default |
| --- | ---: |
| `--cluster-signal-center-bonus` | `0.08` |
| `--cluster-signal-route-bonus` | `0.06` |
| `--cluster-signal-lost-penalty` | `0.08` |
| `--cluster-signal-max-abs` | `0.20` |

Reasoning:

- reward clusters only when our agent is not slower to the selected target
- reward route value because good cluster play often means the next item is
  nearby, not only the current item
- penalize chasing dense-looking targets when the opponent is clearly favored
  to arrive first
- keep the signal clipped so it guides target selection without dominating
  the final score objective

The log reports this through the existing field:

```text
mean_shaping=...
```

## Opponent Pool

Training opponent mix:

```text
bfs:0.15
rollout_gated:0.25
ppo:0.25
position_aware_ppo:0.35
```

The `ppo` and `position_aware_ppo` opponents are frozen checkpoints:

```text
../checkpoints/target_selection_ppo_mlp_abandon_30_latest_best.pt
../checkpoints/target_selection_ppo_mlp_position_selfplay_latest_best.pt
```

They do not update during training.

## Training Run

Default v2 training settings:

```text
iterations=30
rollout_steps=6000
ppo_epochs=4
batch_size=512
hidden_dim=128
learning_rate=2.5e-4
entropy_coef=0.02
terminal_win_bonus=5.0
```

Command from this folder:

```powershell
python .\train_target_selection_ppo_v2.py --iterations 30 --rollout-steps 6000 --ppo-epochs 4 --batch-size 512 --hidden-dim 128 --learning-rate 2.5e-4 --entropy-coef 0.02 --terminal-win-bonus 5.0 --reward-mode score_delta_cluster --opponent-mix bfs:0.15,rollout_gated:0.25,ppo:0.25,position_aware_ppo:0.35 --eval-opponents bfs rollout_gated ppo position_aware_ppo --selection-opponents bfs rollout_gated ppo position_aware_ppo --eval-seeds 0 1 2 3 4 --checkpoint-path .\checkpoints\target_selection_ppo_mlp_128_v2_cluster_latest.pt
```

Watch command:

```powershell
Get-Content C:\pythonlek\8sem\Reinforcement-Learning-\final_project\finalproject\part2\deeprl_v2\training_logs\target_selection_ppo_mlp_128_v2_cluster_stdout.log -Wait -Tail 40
```

## Success Criterion

Do not package this model unless it beats the position-aware PPO on a 10-seed
eval against the real deterministic baselines, while staying competitive
against both frozen PPO opponents.

Current target to beat:

```text
position-aware PPO:
BFS 0..9:           +9.00
rollout_gated 0..9: +6.70
```

## League Self-Play Loop

`league_train_v2.py` automates the next step:

```text
train candidate
evaluate best checkpoint on 10 seeds
accept only if it improves enough
archive accepted checkpoint
append accepted checkpoint to future training opponent pool
repeat
```

This is frozen-snapshot self-play, not live self-play. Once a candidate is
accepted, it becomes a fixed opponent through the generic `league` opponent.
The active learner never updates the opponents.

Default accept gate:

```text
gate_opponents=bfs, rollout_gated, ppo, position_aware_ppo
deterministic_gate_opponents=bfs, rollout_gated
ppo_gate_opponents=ppo, position_aware_ppo
initial_best_score=6.70
improve_margin=0.50
ppo_gate_min_score=4.00
```

A candidate must satisfy both:

```text
min(BFS_mean, rollout_gated_mean) >= previous_best + 0.50
min(PPO_mean, position_aware_PPO_mean) >= 4.00
```

With the current defaults, the deterministic part starts as:

```text
min(BFS_mean, rollout_gated_mean) >= 7.20
```

The PPO part ensures a new candidate is not accepted just because it beats the
deterministic baselines while regressing badly against the two frozen PPO
variants.

When at least one model has been accepted, later generations reserve `0.20`
of the training opponent probability for the accepted league snapshots. The
original opponent mix is scaled down to make room.

Run one automated generation:

```powershell
python .\league_train_v2.py --generations 1
```

Run three automated generations:

```powershell
python .\league_train_v2.py --generations 3
```

Main outputs:

```text
league_pool.json
checkpoints/league_genXX_latest_best.pt
checkpoints/league_genXX_accepted.pt
training_logs/league_genXX_train.log
training_logs/league_genXX_eval10.log
```

Yes: each generation gets its own training log and its own 10-seed evaluation
log. For example, generation 1 writes:

```text
training_logs/league_gen01_train.log
training_logs/league_gen01_eval10.log
```

The accepted pool can be inspected in `league_pool.json`. If a candidate does
not pass the gate, it is not appended to the league pool, but its logs and
checkpoints remain available for analysis.
