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

## 2026-05-06 Route-Feature Run

The first simple MLP PPO run learned a policy that was somewhat competitive
against BFS but still clearly worse than `rollout_gated`. The pattern looked
like convergence to a non-superior policy rather than numerical instability:

- eval against BFS became mildly positive on some seeds
- eval against `rollout_gated` stayed negative
- entropy fell steadily, meaning the policy became confident
- KL stayed small/moderate, so the update was not exploding

Interpretation: the simplified model had enough information to learn a
reasonable nearest-item/anti-BFS policy, but not enough information to discover
the route-aware behavior that made `rollout_gated` strong.

Implemented changes:

- Added one route-value feature to each candidate target.
- Changed the default training reward to score-difference delta.
- Increased default entropy regularization from `0.01` to `0.02`.
- Changed the default opponent mix from `bfs:0.5,rollout_gated:0.5` to
  `bfs:0.25,rollout_gated:0.75`.
- Changed the default checkpoint name to avoid mixing incompatible feature
  dimensions with the previous MLP checkpoint.

### Route-Value Feature

The route-value feature is deliberately small and deterministic. For a
candidate target, it greedily estimates whether taking that target opens a
short path through nearby follow-up items:

```text
target -> nearest remaining item -> nearest remaining item -> ...
```

The value is distance-discounted and normalized before being given to the MLP.
This does not reintroduce the old CNN architecture or behavior cloning. It
only gives the policy access to the key signal that the deterministic
`rollout_gated` agent used well: a slightly farther item can be better if it
leads into a dense route.

### Score-Delta Reward

The environment reward includes movement penalties, so reward difference and
final score difference are not identical. For target selection, the metric we
care about is item score:

```text
score_diff = own_items - opponent_items
```

The new default training reward is:

```text
new_score_diff - previous_score_diff
```

This means PPO reinforces target choices that directly improve the item-score
race. The raw environment reward can still be used with:

```powershell
--reward-mode env_diff
```

### Stronger Opponent Pressure

The previous run improved more against BFS than against `rollout_gated`.
The first route-feature run sampled `rollout_gated` more often:

```text
bfs:0.25,rollout_gated:0.75
```

The intent is to stop the policy from settling for behavior that only beats
nearest-item BFS.

### Result

The route-feature run selected iteration 14 as the best balanced checkpoint:

```text
checkpoints/target_selection_ppo_mlp_route_latest_best.pt
```

Training eval at iteration 14:

| Opponent | Seeds | Mean score diff | Min | Max |
| --- | --- | ---: | ---: | ---: |
| BFS | `0..4` | `-1.00` | `-12` | `+13` |
| rollout_gated | `0..4` | `+1.20` | `-2` | `+8` |

Follow-up eval over 10 seeds:

| Opponent | Seeds | Mean score diff | Min | Max |
| --- | --- | ---: | ---: | ---: |
| BFS | `0..9` | `+0.90` | `-12` | `+13` |
| rollout_gated | `0..9` | `+1.10` | `-3` | `+8` |

This is the first simple-MLP checkpoint that is positive against both BFS and
`rollout_gated` over 10 seeds, but the margin is small. It is not yet a strong
submission replacement. It should be treated as a promising baseline for the
next experiment rather than a final model.

Recorded replays:

- `recordings/ppo_route_best_iter14_vs_bfs_seed0.json`
- `recordings/ppo_route_best_iter14_vs_rollout_gated_seed0.json`

Next likely direction:

- keep the route feature
- use a less skewed opponent mix such as `bfs:0.4,rollout_gated:0.6`
- train/evaluate over more seeds
- save by the minimum of BFS and rollout_gated eval, as currently done
- consider a small parameter sweep over entropy and opponent mix before adding
  architecture complexity

### Active Follow-Up Run

Started a follow-up run with less skew toward `rollout_gated`:

```text
iterations=20
opponent_mix=bfs:0.4,rollout_gated:0.6
reward_mode=score_delta
entropy_coef=0.02
checkpoint=checkpoints/target_selection_ppo_mlp_route_mix046_latest.pt
log=training_logs/target_selection_ppo_mlp_route_mix046_stdout.log
```

Reasoning:

- The `bfs:0.25,rollout_gated:0.75` run found a checkpoint that was mildly
  positive against both opponents over 10 seeds.
- Its final iterations drifted and became weaker against BFS, so 30 iterations
  was probably more than needed for this setup.
- A 20-iteration run is cheaper and should reduce late-run drift while still
  giving PPO time to improve.
- The `0.4/0.6` mix should keep enough pressure from `rollout_gated` while
  reducing the chance that the policy gives up simple BFS races.

Early log snapshot:

- iteration 1 was very poor against both opponents
- iteration 2 improved sharply but was still negative
- iteration 4 was the best early balanced checkpoint so far:
  - BFS mean score diff `-1.80`
  - rollout_gated mean score diff `-3.20`

This run was stopped after iteration 9 so we could test the richer-critic
hypothesis instead of spending more time on a value function that looked too
weak.

### Richer Critic Run

Stopped the `mix046` run early to test the hypothesis that the value function
was not seeing enough state information. The actor was already receiving
per-target features, but the critic only had:

```text
mean(target_features), max(target_features), min(target_features), candidate_count
```

That loses important structure. The critic may not reliably estimate whether a
state is good, which makes PPO advantages noisy even if the actor features are
reasonable.

Implemented change:

- keep the actor unchanged as a shared per-target MLP
- add a separate 14-dimensional global state vector for the critic
- concatenate the global state vector with the pooled target summaries
- use a new checkpoint name because old checkpoints are shape-incompatible

New critic input:

```text
mean(target_features),
max(target_features),
min(target_features),
global_state_features,
candidate_count
```

The global state features include:

- own and opponent positions
- remaining item count
- min and mean own distance to item targets
- min and mean opponent distance to item targets
- fraction of contested targets
- fraction of targets where we are not slower than the opponent
- best route value
- best local cluster counts

Reasoning:

- the policy should still be interpretable as target selection
- BFS still handles movement
- the actor does not get a board CNN or a larger action space
- the critic gets enough context to reduce noisy/incorrect advantage estimates

New run command shape:

```text
iterations=20
opponent_mix=bfs:0.4,rollout_gated:0.6
reward_mode=score_delta
entropy_coef=0.02
checkpoint=checkpoints/target_selection_ppo_mlp_route_critic_latest.pt
log=training_logs/target_selection_ppo_mlp_route_critic_stdout.log
```

Early richer-critic result:

- the value-function input became richer, but the actor still lacked explicit
  information for abandoning losing races
- early eval improved from extremely poor iteration 1 results, but still
  showed large losses and instability
- this run was stopped after iteration 5 to add direct race-abandonment and
  score-context inputs

### Abandonment-Aware Target Features

Replay inspection showed the policy often kept chasing targets it was likely
to lose to the opponent. The policy can technically switch target every step,
so the issue was not mechanical commitment. The issue was that the actor did
not get sufficiently direct input saying:

```text
this target is contested
the opponent is clearly favored
we are behind and should stop wasting steps
```

Implemented changes:

- target feature dimension increased from `14` to `23`
- global critic feature dimension increased from `14` to `19`
- added score and time context from the observation
- added per-target race-abandonment features
- changed checkpoint naming again because old checkpoints are
  shape-incompatible

New per-target actor features:

- contested target flag
- own-favored race flag
- lost-race flag
- route value if own-favored
- route value if race is already lost
- normalized score difference
- behind flag
- late-game behind pressure
- abandon pressure for losing races

New global critic features:

- normalized score difference
- behind flag
- current step progress
- remaining step fraction
- late-game behind pressure

Reasoning:

- route value should not make a target look good if the opponent will reach it
  first
- if the agent is behind, spending steps on losing races is especially costly
- the actor now gets explicit signals for abandoning bad targets instead of
  having to infer that behavior indirectly from distances and sparse rewards

New run command shape:

```text
iterations=20
opponent_mix=bfs:0.4,rollout_gated:0.6
reward_mode=score_delta
entropy_coef=0.02
checkpoint=checkpoints/target_selection_ppo_mlp_abandon_latest.pt
log=training_logs/target_selection_ppo_mlp_abandon_stdout.log
```

Status before leaving school:

- run was started and then stopped after iteration 1
- iteration 1 was still an untrained/random-policy checkpoint, so it is not
  useful to preserve as a model result
- restart this experiment from scratch when training can run uninterrupted

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
bfs:0.4,rollout_gated:0.6
```

Default checkpoint:

```text
checkpoints/target_selection_ppo_mlp_abandon_latest.pt
```

Recommended training command from `final_project/finalproject/part2`:

```powershell
python .\train_target_selection_ppo.py --iterations 20 --rollout-steps 3000 --ppo-epochs 4 --batch-size 256 --learning-rate 2.5e-4 --entropy-coef 0.02 --reward-mode score_delta --opponent-mix bfs:0.4,rollout_gated:0.6 --eval-opponents bfs rollout_gated --eval-seeds 0 1 2 3 4 --checkpoint-path .\checkpoints\target_selection_ppo_mlp_abandon_latest.pt
```

Evaluate a checkpoint:

```powershell
python .\train_target_selection_ppo.py --eval-only --checkpoint-path .\checkpoints\target_selection_ppo_mlp_abandon_latest_best.pt --eval-opponents bfs rollout_gated --eval-seeds 0 1 2 3 4 5 6 7 8 9 --eval-steps 1000
```

Record a replay:

```powershell
python .\record_target_selection_ppo.py --checkpoint-path .\checkpoints\target_selection_ppo_mlp_abandon_latest_best.pt --opponent bfs --seed 0 --output .\recordings\target_selection_ppo_mlp_abandon_vs_bfs_seed0.json
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

## ojay, notes. 
- think 2 layer fc mlp approach looks promising. 
- can think about enriching input if neccessary, more feature eng perhaps. Also, parameter search as in the 2017 OpenaAi paper at some point?.

## want next updates under here!
