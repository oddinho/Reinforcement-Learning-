
 ## REPORT ON PPO

 # ex6

 ## intro 
 1. challenges and limitations in the field, detected by authors: 
 - the reasoning was that current soluton methods or approaches was either not understandable, scalable or sample efficient. Scaling to larger models would need a more understandable and efficient algorithm, which was the motivation for PPO. With efficiency, one typically refers to use of first order information(derivatives) instead of second order derivatives. Scalability here also implies the possibility to parallelize training. 
 2. aim? 
 - The authors simply want to achieve results comparable or better than Trust Region Policy Optimization (TRPO), which uses second order information, using only first order information. 
 3. contributions? 
 - Introduces PPO as a new family of policy-gradient algorithms, which keep many of the benefits of TRPO but only uses first order information .
 4. eval?
 - to acess the quality of their results, the authors compare the new algorithm(ppo)'s performance against other(previous) algorithms on different tasks. Example is continous control tasks, or on atari vs A2C and ACER(actor-critic methods). 
 ## trpo related
 10. Surrogate objective function? 
 - The meaning and role of a surrogate objective function is to give us a practical oprimization target for improving the policy. (. expand if neccessary)
 11. meaning of \pi_{\theta}_{old}? 
 - the policy pi_old is the previous vector of policy parameters, and acts as a reference for our new updated vector of policy parameters. This is to avoid or penalize too large changes, as we wish to maximise this ratio times the estimated advantage. 
 ## trpo v ppo
 21. relation between eq(7) and eq(3-4) ? 
 - the relation between equation 7 and equation 3 and 4 is mostly simplyfying and or slightly changing the objective and constraints. Equation 7 is PPO's simplified version of the TRPO objective in eq3, 4. Both versions include the probability ratio r_t(\theta) times the estimated advantage. The difference is that PPO simply clips the ratio onto the interval [1-\epsilon, 1+\epsilon] instead of a KL constraint/penalty. 
 22. 
 - On the y axis you have one timestep of the surrogate function L^{clip}, and on the x axis you have r_t. Further, the figure shows how once the probability ratio reach the boundries of the chosen interval, it is clipped onto the interval, which removes the incentive to make too large of a policy update. 
 23. 
 - There are two plots because the clipping works differently depending on if A(advantage) is positive/negative. If advantage is positive, want to increase prob of sampled action, if negative, decrease prob sampled action. Since r_t > for all \theta, L^{clip} > 0 for A >0, and L^{clip} < 0 for A <0. 
 24. 
 -  The red circle is the starting point for the optimization, in other words where r_t = 1. This means one assumes that the new policy (\pi_{\theta}) assigns the same probability to the new sampled action as the old policy. 
## main obj function, different parts
 37. 
 - role of each term in making up the overall loss? The first term is the one we are already familiar with, the policy improvement term. further, the objective incluldes a term which for the squared error loss for the value function, which is subtracted form the policy improvement term in the main obj function which one wishes to maximize. Finally, S represents an entropy bonus, to ensure some exploration happens.
 ## comp. surrogate obj part.
 46. 
 - which env? 7 simulated robotics tasks from OpenAI Gym: HalfCheetah-v1, Hopper-v1, InvertedDoublePendulum-v1, InvertedPendulum-v1, Reacher-v1, Swimmer-v1, Walker2d-v1. Action sets: continous action vectors. 
 47. 
 - hyperparams? \epsilon, \beta, d_targ. + T: horizon, adam stepsize = lr, num epochs = how many times same collected batch is reused, minibatch size, \gamma = discount factor future rewards, and \lambda, "GAE param controlling bias-variance tradeoff in adv. estimateion". 
 48. 
 - searched over: all surrogate hyperparams: epsilon, beta, d_targ
 - fixed: the other ppo trainin hyperparams: T, lr(adam stepsize), num epochs, minbatch size, gamma and lamda. 
 49. 
 -why no c1, c2. They did not share hyperparameters between the policy and value function, so value function coeff irrelevant(c1). They also did not use an entropy bonus, so c2 irrelevant. 
 50. 
 - the reason for fixing some of the hyperparams are that it would be cheaper computationally, and also makes the experiment comparable. Note, point of 6.1 was to compare surrogate obj variants. 
## architecture
 51. 
 - NN with 2 fully connected layers, tanh activation, output gaussian. (draw fully connected mlp)
 52. 
 - the output is a probability distribution over possible actions. One outputs a gasussian, with mean and std.dev, and this is so that the model can handle continous action space. Having a probability dist over cont. space handles choosing actions. 
 53. 
 -  why average total reward over the last 100 episodes? Want a measure of final performance after training, and not so much results during earlier learning stages. 
 54. 
 - score normalization: They normalize scores separately for each env, by shifting and scaling score s.t. random policy gets score = 0 and best result gets score = 1. They then take the average across 21 runs to get one scalar score for each algo. 
 55. why avg over 21 runs?
 - 3 runs per env. (7 envs, 3 random seeds on each env).
 56. 
 - no scores = 1 follow directly from best run being set as score1. And as all vectors are averages, they will all contain runs which performed worse than the best run, and therefore have a lower score than 1. 
 57. 
 - negative values appear because for some hyperparmam setting for the surrogate obj, the optimization algo performed worse than random choice. 
## main conclusions:
 58. 
 - The main conclusion is that PPO provides a simpler, fist order alternative to TRPO, while keeping stable and reliable policy updates. The authors note its strong performance in the experiments, and how it offers a good balance between sample efficiency, simplicity and performance. 
 