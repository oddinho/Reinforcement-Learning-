import numpy as np
import matplotlib.pyplot as plt

from bandit import Bandits_one

SEED = 42

def epsilon_gready(MAB,time=1000,epsilon=0.1, seed= SEED):
    rng = np.random.default_rng(seed)

    Q = np.zeros(MAB.k)   # estimated rewards
    N = np.zeros(MAB.k)   # action counts
    
    rewards = np.zeros(time)
    actions = np.zeros(time, dtype=int)


    for t in range(time):
        # sample e ~ Uniform[0,1]
        if rng.random() < epsilon:
            a = rng.integers(MAB.k)     # random action
        else:
            a = np.argmax(Q)            # greedy action

        _, r, _, _, _ = MAB.step(a)

        # update
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]

        actions[t] = a
        rewards[t] = r

    return Q, N, actions, rewards












