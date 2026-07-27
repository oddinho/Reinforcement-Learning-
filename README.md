# Reinforcement Learning Coursework and Collector Agent

An academic repository covering reinforcement-learning theory, practical
implementations, and a competitive final project. The work progresses from
multi-armed bandits and dynamic programming to temporal-difference learning,
planning, value-function approximation, policy gradients, actor-critic methods,
and Proximal Policy Optimization (PPO).

## Final project highlight

**Our PPO-based agent placed 2nd among approximately 20 teams in the course's
final Collector Gridworld competition.**

Collector is a two-player gridworld in which agents navigate around obstacles
and compete to collect items. The final approach separates strategic target
selection from movement:

```text
grid observation
      |
reachable item candidates + engineered features
      |
PPO actor scores and selects a target
      |
BFS converts the target into a shortest-path action
```

This hybrid design lets the learned policy focus on the competitive decision:
which item or cluster to pursue. Deterministic pathfinding handles navigation.

The PPO experiments include:

- compact target features based on path distance, item density, route value,
  score state, and the race against the opponent;
- clipped PPO updates with a learned value function, entropy regularisation,
  and Generalised Advantage Estimation;
- deterministic BFS and rollout-based agents as training and evaluation
  baselines;
- opponent pools and frozen PPO snapshots for more robust self-play;
- fixed-seed evaluation and checkpoint selection; and
- replay generation for qualitative inspection of agent behaviour.

The PPO implementation uses PyTorch and NumPy without an external RL training
library.

## Coursework

| Area | Topics | Main notebook |
| --- | --- | --- |
| Foundations | Multi-armed bandits and exploration strategies | [`notebooks/homework1/final_notebook.ipynb`](notebooks/homework1/final_notebook.ipynb) |
| Markov decision processes | Markov reward processes, Bellman equations, optimality, and dynamic programming | [`notebooks/homework2/a2_ojay.ipynb`](notebooks/homework2/a2_ojay.ipynb) |
| Model-free learning | Monte Carlo methods, importance sampling, Sarsa, and Q-learning | [`notebooks/homework3/a3_ojay.ipynb`](notebooks/homework3/a3_ojay.ipynb) |
| Planning | Function approximation, Monte Carlo Tree Search, and Dyna | [`notebooks/homework4/ojay4.ipynb`](notebooks/homework4/ojay4.ipynb) |
| Deep RL | Policy gradients, actor-critic methods, replay buffers, and target networks | [`notebooks/homework5/assignment5.ipynb`](notebooks/homework5/assignment5.ipynb) |

## Repository structure

```text
.
|-- notebooks/                          # Five coursework assignments
|-- src/                                # Earlier custom environments
`-- final_project/
    |-- part1/                          # Written project work
    |-- part3.md                        # PPO paper review
    `-- finalproject/
        |-- src/
        |   |-- environments/collector/ # Competitive gridworld
        |   |-- agents/                 # Agent interface and baselines
        |   |-- compete/                # Match runner
        |   `-- visualiser/             # Interactive play and replay tools
        `-- part2/                       # BFS, rollout, DQN, and PPO experiments
```

Detailed environment documentation and examples are available in the
[`final_project/finalproject` README](final_project/finalproject/README.md).
The target-selection PPO design is documented in
[`part2/deeprl_v1/target_selection_ppo.md`](final_project/finalproject/part2/deeprl_v1/target_selection_ppo.md).

## Run a Collector match

From the repository root:

```bash
cd final_project/finalproject
python -m pip install -e .
python -m pip install matplotlib pyyaml
python src/compete/compete.py src/agents/baseline/ src/agents/random/ --output replay.json
python src/visualiser/visualiser.py replay.json
```

To play the environment manually:

```bash
python src/visualiser/play_collector.py 0
```

The deep-RL experiments additionally require PyTorch and NumPy.

## Collaboration note

This repository was shared by three students for coursework and discussion.
Some individual assignment work was maintained on separate branches, while the
final-project environment and agent development were collaborative.
