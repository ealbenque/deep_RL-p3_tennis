## Report : Udacity deep RL nanodegree - project 3

### Implementation
The resolution of the environment involves the utilization of a deep reinforcement learning agent.
* tennis.ipynb contains the main code with the training loop and results
* agent.py contains the reinforcement learning agent
* model.py includes the neural networks serving as the estimators
* the weights of the different neural networks are saved in separate files.

### Learning Algorithm
The environnement is solved using a multi-agent DDPG algorithm [MADDPG] (https://arxiv.org/abs/1706.02275). It uses two agents with a shared replay buffer. 

DDPG, or Deep Deterministic Policy Gradient, is a reinforcement learning algorithm designed for solving continuous action space problems. It combines ideas from deep learning and policy gradient methods to handle high-dimensional state and action spaces. Here's a brief explanation of the key components and concepts of DDPG:
* DDPG employs an actor-critic architecture, where the actor is responsible for learning a deterministic policy, mapping states to specific actions, and the critic evaluates the state-action pairs.
* Deterministic Policy: the actor in DDPG learns a deterministic policy, meaning it directly outputs the action to be taken given a particular state. This is in contrast to stochastic policies that output a probability distribution over actions.
* Experience Replay Buffer: DDPG uses an experience replay buffer to store past experiences like in DQN algorithms (tuples of state, action, reward, next state) for training. This helps in breaking the temporal correlations in the data and provides more stable learning.
* Target Networks: to stabilize training, DDPG uses target networks for both the actor and the critic. These are copies of the original networks that are slowly updated over time using a soft update mechanism.
* Q-function Approximation: the critic approximates the action-value function (Q-function), which estimates the expected cumulative reward of taking a specific action in a given state and following a particular policy.
* Policy Gradient: DDPG uses the policy gradient method to update the actor network. The gradient is computed with respect to the expected return and is used to adjust the actor's parameters in the direction that increases the expected return.
* Target Q-value: The target Q-value is used to update the critic network. It is computed using the Bellman equation and is used as a target for the critic's Q-value prediction.
By combining these elements, DDPG is able to learn a deterministic policy for continuous action spaces.
MADDPG extends DDPG to multi-agents domains.

### Hyperparameters
* Buffer size : 1e6
* Replay batch size (shared) : 512
* Update frequency : every 4
* discount factor : 0.99
* Soft update parameter TAU : 1e-3
* Actor learning rate : 1e-4
* Critic learning rate : 3e-4
* Dropout for critic network : 20 %
* Ornstein-Uhlenbeck noise : mu=0., theta=0.15, sigma=0.2

### Neural networks
The actor model is a simple feedforward network: maps state to action
* Input layer: 33  neurons (the state size)
* 1st hidden layer: fully connected, 512 neurons with ReLu activation
* 2nd hidden layer: fully connected, 256 neurons with ReLu activation
* output layer: 4 neurons (1 for each action) (tanh)

The critic model: maps state action pairs to value
* Batch normalization
* Input layer: 33 neurons (the state size) + 4 neurons (1 for each actions)
* 1st hidden layer: fully connected, 512 neurons with ReLu activation
* 2nd hidden layer: fully connected, 256 neurons with ReLu activation
* output layer: 1 neuron

### Plot of Rewards
Plot of rewards can be seen after the environment has been solved.
Environment solved after 100 episodes.
![plot](https://github.com/ealbenque/deep_RL-p3_tennis/assets/137990986/e1557afc-76a1-40c3-aee9-7d1c1dd29eda)

### Possible improvements
- PPO to address issues related to policy optimization instability
- fine tuning hyperparameters for an agent
- Prioritized experience replay
