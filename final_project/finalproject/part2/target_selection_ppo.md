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

Each reachable item target gets a compact 23-dimensional feature vector:

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

## Training

From `final_project/finalproject/part2`:

```powershell
python .\train_target_selection_ppo.py --iterations 20 --rollout-steps 3000 --ppo-epochs 4 --batch-size 256 --learning-rate 2.5e-4 --entropy-coef 0.02 --reward-mode score_delta --opponent-mix bfs:0.4,rollout_gated:0.6 --eval-opponents bfs rollout_gated --eval-seeds 0 1 2 3 4 --checkpoint-path .\checkpoints\target_selection_ppo_mlp_abandon_latest.pt
```

The script saves:

- latest checkpoint: `--checkpoint-path`
- best checkpoint: same name with `_best.pt`

The best checkpoint is selected by the weakest mean score difference across
the configured eval opponents. This prevents selecting a model that only beats
one opponent.

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
