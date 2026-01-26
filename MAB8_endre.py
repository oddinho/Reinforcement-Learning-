import numpy as np
import matplotlib.pyplot as plt

from bandit import Bandits_one

SEED = 42

def epsilon_gready(MAB,time=1000,epsilon=0.1, seed= SEED):
    rng = np.random.default_rng(seed)

    Q = np.zeros(MAB.k)   # estimated rewards
    N = np.zeros(MAB.k)   # action counts







def epsilon_greedy(env, T=1000, eps=0.1, seed=0):
    """
    Simple epsilon-greedy for a K-armed bandit.

    env: bandit environment with env.k and env.step(a)
    T:   time horizon
    eps: exploration probability
    """


    rewards = np.zeros(T)
    actions = np.zeros(T, dtype=int)

    for t in range(T):
        # sample e ~ Uniform[0,1]
        if rng.random() < eps:
            a = rng.integers(env.k)     # random action
        else:
            a = np.argmax(Q)            # greedy action

        _, r, _, _, _ = env.step(a)

        # update
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]

        actions[t] = a
        rewards[t] = r

    return Q, N, actions, rewards
