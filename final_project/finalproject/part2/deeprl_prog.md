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

### Restarted 30-Iteration Abandonment Run

Restarted the abandonment-aware PPO experiment from scratch after getting back
to a machine that can run uninterrupted.

Current run:

```text
iterations=30
rollout_steps=3000
ppo_epochs=4
batch_size=256
learning_rate=2.5e-4
entropy_coef=0.02
reward_mode=score_delta
opponent_mix=bfs:0.4,rollout_gated:0.6
eval_opponents=bfs, rollout_gated
eval_seeds=0,1,2,3,4
checkpoint=checkpoints/target_selection_ppo_mlp_abandon_30_latest.pt
log=training_logs/target_selection_ppo_mlp_abandon_30_stdout.log
```

The reason for using 30 iterations here, instead of the earlier 20-iteration
budget, is that this is now a changed state representation rather than just a
small opponent-mix tweak. The policy has new direct information about contested
targets, lost races, score pressure, and late-game urgency. It is reasonable to
give PPO more room to adapt before judging whether these features help.

First logged checkpoint:

| Iteration | BFS mean | rollout_gated mean | Selection score |
| --- | ---: | ---: | ---: |
| 1 | `-1.00` | `-6.00` | `-6.00` |

Interpretation of iteration 1:

- it is not yet evidence that the abandonment features failed
- the rollout training return was still very negative, which is expected for a
  near-random early policy
- BFS eval was already close to even on five seeds, but `rollout_gated` was
  still clearly ahead
- the useful question is whether later iterations improve the worst opponent,
  not whether iteration 1 beats the deterministic agents

What to watch:

- the saved-best selection score should improve above `-6.00`
- BFS mean should stay near zero or positive while rollout_gated improves
- entropy should not collapse too early
- KL should remain nonzero but moderate; near-zero KL for many iterations would
  mean the policy is barely changing
- replay inspection should specifically check whether the agent abandons
  targets when the opponent is closer and the route value is no longer useful

Completed result:

- the run completed all 30 iterations
- stderr was clean
- final checkpoint was not the best checkpoint
- the saved-best checkpoint was iteration 19

Best checkpoint:

```text
checkpoints/target_selection_ppo_mlp_abandon_30_latest_best.pt
```

Training eval for the saved-best checkpoint:

| Opponent | Seeds | Mean score diff | Min | Max |
| --- | --- | ---: | ---: | ---: |
| BFS | `0..4` | `+11.20` | `+3` | `+18` |
| rollout_gated | `0..4` | `+7.20` | `-4` | `+14` |

Follow-up eval for the saved-best checkpoint:

| Opponent | Seeds | Mean score diff | Min | Max |
| --- | --- | ---: | ---: | ---: |
| BFS | `0..9` | `+9.90` | `+3` | `+18` |
| rollout_gated | `0..9` | `+4.70` | `-4` | `+14` |

Eval logs:

- `training_logs/target_selection_ppo_mlp_abandon_30_best_eval10.log`
- `training_logs/target_selection_ppo_mlp_abandon_30_best_eval10_rollout_gated.log`

Final checkpoint at iteration 30:

| Opponent | Seeds | Mean score diff | Min | Max |
| --- | --- | ---: | ---: | ---: |
| BFS | `0..4` | `+6.00` | `-1` | `+13` |
| rollout_gated | `0..4` | `-3.40` | `-13` | `+6` |

Interpretation:

- the abandonment-aware features helped substantially compared with the
  earlier route-only MLP runs
- iteration 19 is much stronger and more balanced than the final checkpoint,
  so late training still drifts
- this reinforces saving by balanced eval instead of using the last checkpoint
- entropy was low by the late iterations, so the policy may be becoming too
  deterministic after it finds a good target-selection style
- the next serious validation step is a wider fixed-seed eval before treating
  this as a candidate submission model

Recorded replays from the iteration-19 best checkpoint:

- `recordings/target_selection_ppo_mlp_abandon_30_best_vs_bfs_seed0.json`
- `recordings/target_selection_ppo_mlp_abandon_30_best_vs_rollout_gated_seed0.json`

Replay note:

- seed 0 vs BFS wins `148-141`
- seed 0 vs rollout_gated loses narrowly `155-159`
- this is consistent with the eval table: rollout_gated seed 0 was the one
  losing seed for the best checkpoint, while seeds 1-4 were positive

Submission package snapshot:

The iteration-19 best checkpoint has been stored locally in upload-ready
structure:

```text
submission_agents/target_selection_ppo_abandon_30_agent/
  agent.py
  config.yaml
  weights/model.pth
```

Clean zip:

```text
submission_agents/target_selection_ppo_abandon_30_agent.zip
```

The zip contains only:

```text
agent.py
config.yaml
weights/model.pth
```

The package was smoke-tested through `src/compete/compete.py` against the
baseline agent and loaded successfully.

### Fine-Tune From Best Checkpoint

Started a short fine-tune from the iteration-19 best checkpoint to see whether
lower learning rate and lower entropy can improve the already-good policy
without causing the late-run drift seen in the 30-iteration run.

Fine-tune setup:

```text
resume_path=checkpoints/target_selection_ppo_mlp_abandon_30_latest_best.pt
iterations=10
rollout_steps=3000
ppo_epochs=4
batch_size=256
learning_rate=1.0e-4
entropy_coef=0.01
reward_mode=score_delta
opponent_mix=bfs:0.4,rollout_gated:0.6
eval_opponents=bfs, rollout_gated
eval_seeds=0,1,2,3,4
checkpoint=checkpoints/target_selection_ppo_mlp_abandon_30_finetune_latest.pt
log=training_logs/target_selection_ppo_mlp_abandon_30_finetune_stdout.log
```

Important caveat:

- this is a fine-tune from model weights, not an exact optimizer-state resume
- the original iteration-19 best checkpoint and submission package are kept
  separately, so the fine-tune cannot overwrite the current best model

Fine-tune result:

- the 10-iteration fine-tune completed cleanly
- the best fine-tune checkpoint was iteration 4 by the same balanced
  5-seed selection score
- it did not improve over the original iteration-19 checkpoint

Fine-tune best checkpoint:

```text
checkpoints/target_selection_ppo_mlp_abandon_30_finetune_latest_best.pt
```

Training eval for the best fine-tune checkpoint:

| Opponent | Seeds | Mean score diff | Min | Max |
| --- | --- | ---: | ---: | ---: |
| BFS | `0..4` | `+5.40` | `-4` | `+15` |
| rollout_gated | `0..4` | `+6.00` | `-2` | `+18` |

Follow-up eval for the best fine-tune checkpoint:

| Opponent | Seeds | Mean score diff | Min | Max |
| --- | --- | ---: | ---: | ---: |
| BFS | `0..9` | `+2.50` | `-12` | `+15` |
| rollout_gated | `0..9` | `+3.90` | `-7` | `+18` |

Comparison to original iteration-19 checkpoint:

| Checkpoint | BFS `0..9` | rollout_gated `0..9` |
| --- | ---: | ---: |
| original iteration 19 | `+9.90` | `+4.70` |
| fine-tune best | `+2.50` | `+3.90` |

Interpretation:

- fine-tuning from the best weights caused policy drift instead of useful
  refinement
- lowering LR and entropy was not enough to preserve the BFS behavior
- keep the original iteration-19 submission package as the current best model
- further improvement should use a fresh run with better checkpoint selection
  or an explicit constraint/regularizer toward the saved policy, rather than
  naive continuation

### Snapshot Self-Play Run

Implemented frozen-snapshot self-play for the next training experiment.

New trainer support:

- `ppo_snapshot` is now a valid opponent name
- `--self-play-checkpoint` points to the frozen PPO model used as opponent
- `--selection-opponents` controls which eval opponents are used for
  best-checkpoint selection
- the snapshot opponent is loaded as a deterministic target-selection PPO
  policy, so the learner plays against a fixed copy of the current best model

Reasoning:

- this is not live self-play where both policies update
- the opponent is a frozen copy of the iteration-19 checkpoint
- this is more stable and easier to interpret
- BFS and rollout_gated remain in the opponent mix so the learner does not
  overfit only to the snapshot policy
- best-checkpoint selection still uses only BFS and rollout_gated because
  those are the real target opponents

Started a fresh 30-iteration run:

```text
iterations=30
rollout_steps=3000
ppo_epochs=4
batch_size=256
learning_rate=2.5e-4
entropy_coef=0.02
reward_mode=score_delta
opponent_mix=bfs:0.25,rollout_gated:0.45,ppo_snapshot:0.30
self_play_checkpoint=checkpoints/target_selection_ppo_mlp_abandon_30_latest_best.pt
eval_opponents=bfs, rollout_gated, ppo_snapshot
selection_opponents=bfs, rollout_gated
eval_seeds=0,1,2,3,4
checkpoint=checkpoints/target_selection_ppo_mlp_selfplay_snapshot_latest.pt
log=training_logs/target_selection_ppo_mlp_selfplay_snapshot_stdout.log
```

What would count as success:

- better 10-seed validation than the original iteration-19 checkpoint
- especially improving rollout_gated while keeping BFS strongly positive
- no replacement unless it beats:
  - BFS `0..9`: `+9.90`
  - rollout_gated `0..9`: `+4.70`

Watch command:

```powershell
Get-Content C:\pythonlek\8sem\Reinforcement-Learning-\final_project\finalproject\part2\training_logs\target_selection_ppo_mlp_selfplay_snapshot_stdout.log -Wait -Tail 40
```

Overnight result:

- the run completed all 30 iterations
- stderr was clean
- the saved-best checkpoint was iteration 14
- the latest checkpoint is iteration 30

Self-play saved-best checkpoint:

```text
checkpoints/target_selection_ppo_mlp_selfplay_snapshot_latest_best.pt
```

Training eval for iteration 14:

| Opponent | Seeds | Mean score diff | Min | Max |
| --- | --- | ---: | ---: | ---: |
| BFS | `0..4` | `+10.80` | `0` | `+18` |
| rollout_gated | `0..4` | `+5.00` | `-5` | `+19` |
| ppo_snapshot | `0..4` | `+1.20` | `-7` | `+8` |

Latest checkpoint:

```text
checkpoints/target_selection_ppo_mlp_selfplay_snapshot_latest.pt
```

Training eval for iteration 30:

| Opponent | Seeds | Mean score diff | Min | Max |
| --- | --- | ---: | ---: | ---: |
| BFS | `0..4` | `+4.00` | `-1` | `+12` |
| rollout_gated | `0..4` | `+4.40` | `-1` | `+8` |
| ppo_snapshot | `0..4` | `+1.00` | `-3` | `+6` |

Interpretation:

- iteration 30 is interesting because it is more balanced and has fewer severe
  losses on the five eval seeds
- iteration 14 is still stronger by the configured selection metric because it
  keeps BFS much higher while also beating rollout_gated
- snapshot self-play did not obviously beat the previous best PPO checkpoint
  on the small eval set, but it produced a competitive saved-best checkpoint
- before packaging anything from this run, run 10-seed validation for both
  iteration 14 and iteration 30 and compare against the original iteration-19
  abandonment checkpoint

### Shaped Cluster/Front-Run Reward Run

Replay inspection after the first submitted PPO model showed two remaining
weaknesses:

- the policy can still keep chasing targets the opponent is clearly favored to
  win
- cluster and route priority is present in the features, but not always strong
  enough in the learned policy

Implemented a new reward mode:

```text
score_delta_shaped
```

This keeps the original score-difference delta as the base reward and adds
small dense target-selection hints from the selected target row:

```text
reward =
    score_delta
    + own_favored_cluster_bonus
    + own_favored_route_bonus
    + front_run_collect_bonus
    - lost_race_penalty
    - lost_route_penalty
```

Default shaping coefficients:

| Term | Value |
| --- | ---: |
| cluster bonus | `0.02` |
| route bonus | `0.03` |
| front-run collect bonus | `0.05` |
| lost-race penalty | `0.04` |
| lost-route penalty | `0.03` |
| per-step shaping clip | `0.15` |

Details:

- cluster and route bonuses apply only when the target race is own-favored
- front-run bonus applies when we collect and the target looked contested or
  strategically valuable
- lost-race penalties apply when the opponent is clearly favored to reach the
  selected target first
- all shaping is small and clipped so final item score remains the main signal

The new rollout log includes:

```text
mean_shaping=...
```

This should make it easier to see whether the shaping terms are actually
active during collection.

Started a fresh 30-iteration shaped snapshot-self-play run:

```text
iterations=30
rollout_steps=3000
ppo_epochs=4
batch_size=256
learning_rate=2.5e-4
entropy_coef=0.02
reward_mode=score_delta_shaped
opponent_mix=bfs:0.25,rollout_gated:0.45,ppo_snapshot:0.30
self_play_checkpoint=checkpoints/target_selection_ppo_mlp_abandon_30_latest_best.pt
eval_opponents=bfs, rollout_gated, ppo_snapshot
selection_opponents=bfs, rollout_gated
eval_seeds=0,1,2,3,4
checkpoint=checkpoints/target_selection_ppo_mlp_shaped_selfplay_latest.pt
log=training_logs/target_selection_ppo_mlp_shaped_selfplay_stdout.log
```

Success criterion:

- must beat or at least match the submitted PPO on 10-seed validation
- especially should improve rollout_gated while keeping BFS strongly positive
- do not package unless it beats the current submitted PPO:
  - BFS `0..9`: `+9.90`
  - rollout_gated `0..9`: `+4.70`

10-seed validation result:

| Opponent | Seeds | Mean score diff | Min | Max |
| --- | --- | ---: | ---: | ---: |
| BFS | `0..9` | `+5.90` | `-7` | `+15` |
| rollout_gated | `0..9` | `+2.30` | `-5` | `+19` |
| ppo_snapshot | `0..9` | `+2.70` | `-10` | `+36` |

Eval log:

```text
training_logs/target_selection_ppo_mlp_shaped_selfplay_best_eval10.log
```

Interpretation:

- the shaped checkpoint looked strong on the 5-seed training eval, but did not
  generalize to 10 seeds
- it is worse than the submitted PPO checkpoint on both real opponents
- do not package this checkpoint
- the shaping signal was probably too seed-specific or too strong relative to
  the sparse score objective
- position features should not be added on top of this checkpoint as a
  fine-tune candidate; if tested, use a fresh run or migrate from the submitted
  iteration-19 checkpoint instead

### Position-Feature Migration From Best PPO

Next experiment: return to the strongest submitted PPO snapshot and add
explicit target-position features.

Starting checkpoint:

```text
checkpoints/target_selection_ppo_mlp_abandon_30_latest_best.pt
```

This checkpoint is still the best validated PPO so far:

| Opponent | Seeds | Mean score diff |
| --- | --- | ---: |
| BFS | `0..9` | `+9.90` |
| rollout_gated | `0..9` | `+4.70` |

Reasoning:

- the shaped self-play checkpoint looked promising on five seeds but got worse
  on ten-seed validation
- the old best abandonment-aware checkpoint generalizes better, so it is the
  cleaner base model
- position may matter because targets near the center can preserve future
  routing options, while isolated edge targets can waste tempo
- the feature should be input-only for this run, not extra reward shaping, so
  we can isolate whether the model benefits from seeing the signal

Implemented position inputs:

| Feature | Meaning |
| --- | --- |
| `center_score` | high when the target is near the middle of the board |
| `edge_score` | high when the target is near an edge |
| `center_cluster_value` | cluster value weighted by centrality |
| `edge_cluster_value` | cluster value weighted by edge proximity |

The target feature dimension increased from `23` to `27`. The original 23
features are kept in the same order so reward shaping indices and old learned
weights remain meaningful.

Checkpoint migration was added to `train_target_selection_ppo.py`:

- old actor input weights are copied into the first 23 columns
- old critic pooled-feature blocks are copied into the corresponding widened
  mean/max/min blocks
- global critic features and candidate-count weights are copied unchanged
- new position-feature columns start with zero input weight, so the migrated
  model initially behaves like the old best checkpoint

Run setup:

```text
resume_path=checkpoints/target_selection_ppo_mlp_abandon_30_latest_best.pt
iterations=30
rollout_steps=3000
ppo_epochs=4
batch_size=256
learning_rate=1.0e-4
entropy_coef=0.02
reward_mode=score_delta
opponent_mix=bfs:0.25,rollout_gated:0.45,ppo_snapshot:0.30
self_play_checkpoint=checkpoints/target_selection_ppo_mlp_abandon_30_latest_best.pt
eval_opponents=bfs, rollout_gated, ppo_snapshot
selection_opponents=bfs, rollout_gated
eval_seeds=0,1,2,3,4
checkpoint=checkpoints/target_selection_ppo_mlp_position_selfplay_latest.pt
log=training_logs/target_selection_ppo_mlp_position_selfplay_stdout.log
```

Success criterion:

- first compare against the submitted PPO on five-seed training eval
- if promising, run ten-seed validation against BFS and rollout_gated
- do not package unless it improves on the submitted PPO baseline:
  - BFS `0..9`: `+9.90`
  - rollout_gated `0..9`: `+4.70`

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
python .\train_target_selection_ppo.py --iterations 30 --rollout-steps 3000 --ppo-epochs 4 --batch-size 256 --learning-rate 2.5e-4 --entropy-coef 0.02 --reward-mode score_delta --opponent-mix bfs:0.4,rollout_gated:0.6 --eval-opponents bfs rollout_gated --eval-seeds 0 1 2 3 4 --checkpoint-path .\checkpoints\target_selection_ppo_mlp_abandon_30_latest.pt
```

Evaluate a checkpoint:

```powershell
python .\train_target_selection_ppo.py --eval-only --checkpoint-path .\checkpoints\target_selection_ppo_mlp_abandon_30_latest_best.pt --eval-opponents bfs rollout_gated --eval-seeds 0 1 2 3 4 5 6 7 8 9 --eval-steps 1000
```

Record a replay:

```powershell
python .\record_target_selection_ppo.py --checkpoint-path .\checkpoints\target_selection_ppo_mlp_abandon_30_latest_best.pt --opponent bfs --seed 0 --output .\recordings\target_selection_ppo_mlp_abandon_30_vs_bfs_seed0.json
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
