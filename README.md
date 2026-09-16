# Implementation of Q Learning Control Algorithm using Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement

Implement the Q-Learning control algorithm in the Gymnasium FrozenLake-v1 environment. The agent should learn the best actions through repeated interaction with the environment and finally learn a policy that helps it reach the goal while avoiding the holes.

## Software Requirements

Python
Gymnasium
NumPy

## Environment Description

The experiment uses the FrozenLake-v1 environment from Gymnasium.

The environment represents a frozen lake where:

The agent starts from a starting state.
The agent can move in four directions.
Some states are safe frozen areas.
Some states are holes.
One state is the goal.
The objective is to reach the goal state while avoiding the holes.

The program creates the environment using:

env = gym.make("FrozenLake-v1", is_slippery=False)

Here, is_slippery=False makes the environment deterministic, so the selected movement happens without the random slipping effect.

## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm

Create the FrozenLake environment.
Obtain the number of states and actions.
Initialize the Q-table with zeros.
Set the learning parameters.
Start training for a fixed number of episodes.
Reset the environment at the beginning of each episode.
Select an action using the epsilon-greedy method.
Execute the action in the environment.
Receive the reward and next state.
Update the Q-value using the Q-Learning update rule.
Move to the next state.
Continue until the episode terminates.
Reduce epsilon after each episode.
Calculate the state-value function.
Extract the learned policy.
Calculate the average reward over the last 1000 episodes.
Display the results.

## Python Program

```python

# -------------------------------------------------
# Q-Learning Training using Gymnasium FrozenLake
# -------------------------------------------------

import gymnasium as gym
import numpy as np

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Environment parameters
state_size = env.observation_space.n
action_size = env.action_space.n

# Q-table initialization
Q = np.zeros((state_size, action_size))

# Hyperparameters
alpha = 0.8          # Learning rate
gamma = 0.95         # Discount factor
epsilon = 1.0        # Initial exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995

episodes = 10000

# Store rewards
rewards = []

# -------------------------------------------------
# Training
# -------------------------------------------------

for episode in range(episodes):

    state, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            # Explore
            action = env.action_space.sample()
        else:
            # Exploit
            action = np.argmax(Q[state])

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Q-Learning update
        if terminated or truncated:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        Q[state, action] = Q[state, action] + \
            alpha * (target - Q[state, action])

        # Move to next state
        state = next_state

        total_reward += reward

    # Store episode reward
    rewards.append(total_reward)

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


# -------------------------------------------------
# Output
# -------------------------------------------------

np.set_printoptions(precision=3, suppress=True)

print("Final Q-table:")
print(Q)

# -------------------------------------------------
# State-Value Function
# V(s) = max Q(s,a)
# -------------------------------------------------

V = np.max(Q, axis=1)

print("\nEstimated State-Value Function:")
print(V.reshape(4, 4))

# -------------------------------------------------
# Learned Policy
# -------------------------------------------------

actions = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑"
}

policy = np.argmax(Q, axis=1)

print("\nLearned Policy:")

for i in range(4):
    row = []
    for j in range(4):
        state = i * 4 + j
        row.append(actions[policy[state]])
    print(row)

# -------------------------------------------------
# Average reward over last 1000 episodes
# -------------------------------------------------

average_reward = np.mean(rewards[-1000:])

print("\nAverage reward over last 1000 episodes:",
      average_reward)

env.close()







```
---

## Output

Final Q-table:


<img width="374" height="301" alt="Screenshot 2026-09-06 151103" src="https://github.com/user-attachments/assets/7879b103-8f35-433e-a5ea-89cba7712cba" />



Estimated State-Value Function:


<img width="406" height="116" alt="Screenshot 2026-09-06 151347" src="https://github.com/user-attachments/assets/b3222202-04a3-462e-be32-62f7355adfcb" />




Learned Policy:

<img width="383" height="112" alt="Screenshot 2026-09-06 151417" src="https://github.com/user-attachments/assets/8074709b-6b68-4db3-9a34-ae11895d0306" />



Average reward over last 1000 episodes: 

<img width="518" height="37" alt="Screenshot 2026-09-06 151449" src="https://github.com/user-attachments/assets/0044a72d-21b2-4b9c-8fcd-2baf95d19a8e" />

```

```

## Result



```text
Thus, the Q-Learning control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment.

The agent learned an optimal action-value function through repeated interaction with the environment using epsilon-greedy exploration and the Q-Learning update rule.

The learned policy enables the agent to reach the goal while avoiding the holes.

```

```
## Inference

```text

The experiment demonstrates that Q-Learning is an off-policy reinforcement learning algorithm that can learn the optimal action-value function without requiring a model of the environment. Initially, the agent explores different actions randomly. As training progresses, the Q-values are updated and the agent increasingly exploits the learned knowledge. Finally, the agent learns a policy that provides a successful route from the starting state to the goal.

```

---

