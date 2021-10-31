# Deep-Q-Learning-with-an-Atari-game

## Abstract
Apply Deep Q Learning to a game - Breakout-v0 in the OpenAI Gym Atari environments. With the deep Q-learning algorithm, it learns from actions in the ε-greedy policy, then maximizes the total reward.

In a Breakout game, the player has 3 chances to take action. The player must control a platform to move left and right to bounce a ball. When that ball hits a brick, the brick will disappear, and the ball will bounce and get a score. When the player fails to bounce the ball on the platform, then the player loses that round. It aims to get the highest rewards. The different state, which is s ∈ {1, 2, . . . , 210}. The actions for the player taking, a ∈ {'NOOP', 'FIRE', 'RIGHT', 'LEFT'}. The reward is zero on all transitions except those on which the player reaches bricks, when it is plus score based on different colors.

In the training model, we execute 10000 timestep to get 38 sets of episodes reward score, then the reward trends can be shown by plot function in pyplot module of matplotlib library to make a plot of points Timestep, Reward.

## Table of contents
* [Installation](#installation)
* [Algorithm](#algorithm)
* [Implementation](#implementation)
* [Explanation](#explanation)

## Installation
* Clone the repo and cd into it:
```
$ git clone https://github.com/openai/baselines.git
$ cd baselines
```
* Install baselines package:
```
$ pip install -e
```
* Install gym[atari] which also installs ale-py with Gym:
```
$ !pip install tensorflow==2.3.1 gym keras-rl2 gym[atari]
```
## Algorithm
Q-learning is an off-policy reinforcement learning algorithm that seeks to find the best action to take given the current state. The Q-learning function learns from actions that are out of the current policy, like taking random actions, so a policy isn’t needed. It seeks to learn a policy that maximizes the total reward.
![image](https://user-images.githubusercontent.com/71042259/139563552-1e9623df-85d9-4dd7-9bb8-b5df366032ce.png)
	
## Implementation
To run this project using DQN:

* Set up parameter for ε-greedy:
```
total_episodes = 50000        # Total episodes
total_test_episodes = 100     # Total test episodes
max_steps = 99                # Max steps per episode
learning_rate = 0.5           # Learning rate (Alpha)
gamma = 0.9                   # Discounting rate (Gamma)

# Exploration parameters
epsilon = 1.0                 # Exploration rate (Epsilon)
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.01             # Exponential decay rate for exploration prob
```
* Define the model of the Breakout Game, then build it for future testing:
```
def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3,height, width, channels)))
    model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model
```
```
model = build_model(height, width, channels, actions)
```
* Set up the deep Q learning agent and conduct ε-greedy policy in the training model:
```
def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  enable_dueling_network=True, dueling_type='avg', 
                   nb_actions=actions, nb_steps_warmup=1000
                  )
    return dqn
```
```
del model
```
	
## Explanation
The plot() function in pyplot module of matplotlib library is used to make a plot of points Timestep, Reward:

```
scores = dqn.test(env, nb_episodes=50, visualize=True)
print(np.mean(scores.history['episode_reward']))
```
![Unknown](https://user-images.githubusercontent.com/71042259/139563915-0f8c9584-2643-40ca-96a5-444791e46fe5.png)

After training through 10000 timestep, we have not been able to get a very good result because we have not trained more episodes. We need more training to get better and continuous improvement results.
