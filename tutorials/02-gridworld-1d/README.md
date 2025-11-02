# Tutorial 2: 1D GridWorld with Sparse Rewards

Welcome to the second tutorial in our Reinforcement Learning (RL) curriculum. In this tutorial, you will learn about a classic problem in RL: sparse rewards and the credit assignment problem. We will build a custom 1D GridWorld environment using Gymnasium, train an agent to navigate it, and visualize the learned policy.

## What You'll Learn

- **Custom Environment Creation**: How to create a custom RL environment using the Gymnasium library.
- **Sparse Rewards**: Understand what sparse rewards are and why they pose a significant challenge for RL agents.
- **Credit Assignment Problem**: Learn about the difficulty of assigning credit to actions when the reward is delayed.
- **Policy Learning**: Implement a simple policy-based approach to solve the environment.
- **Visualization**: Visualize the agent's learned policy to understand its behavior.

## The 1D GridWorld Environment

Our environment is a simple 1D grid of size 10. The agent starts at the leftmost cell (position 0) and its goal is to reach the rightmost cell (position 9).

- **State**: The agent's position on the grid (an integer from 0 to 9).
- **Actions**: The agent can move left (0) or right (1).
- **Reward**: The agent receives a reward of +1 if it reaches the goal at position 9, and 0 for all other steps.
- **Episode Termination**: An episode ends when the agent reaches the goal.

This setup, with a single non-zero reward at the end, is an example of a **sparse reward** environment.

## Environment Creation

We will create a custom environment by inheriting from `gymnasium.Env`. This requires implementing several methods:

- `__init__()`: To initialize the environment's properties.
- `reset()`: To reset the environment to its initial state.
- `step()`: To execute an action and return the next state, reward, and other information.
- `render()`: To visualize the environment (optional but good practice).

Here is the code for our `GridWorld1D` environment:

```python
import gymnasium as gym
import numpy as np

class GridWorld1D(gym.Env):
    def __init__(self, size=10):
        self.size = size
        self.agent_pos = 0

        self.action_space = gym.spaces.Discrete(2)  # 0: left, 1: right
        self.observation_space = gym.spaces.Discrete(self.size)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = 0
        return self.agent_pos, {}

    def step(self, action):
        if action == 0:  # Move left
            self.agent_pos = max(0, self.agent_pos - 1)
        elif action == 1:  # Move right
            self.agent_pos = min(self.size - 1, self.agent_pos + 1)

        terminated = self.agent_pos == self.size - 1
        reward = 1 if terminated else 0
        truncated = False # Not used in this simple env

        return self.agent_pos, reward, terminated, truncated, {}

    def render(self):
        grid = ['-'] * self.size
        grid[self.agent_pos] = 'A'
        grid[self.size - 1] = 'G'
        print(" ".join(grid))

```

### Explanation of the Code

- **`__init__(self, size=10)`**: We define the size of the grid, the agent's starting position, the action space (`Discrete(2)` for left/right), and the observation space (`Discrete(self.size)` for each position).
- **`reset(self, ...)`**: This method is called at the beginning of each new episode. It resets the agent's position to 0 and returns the initial observation.
- **`step(self, action)`**: This method contains the core logic of the environment. It updates the agent's position based on the chosen action, ensuring the agent stays within the boundaries (`max(0, ...)` and `min(self.size - 1, ...)`). It then checks if the agent has reached the goal (`terminated`). The reward is 1 if the goal is reached, otherwise 0. It returns the new state, reward, terminated flag, truncated flag, and an info dictionary.
- **`render(self)`**: This method provides a simple text-based visualization of the environment, showing the agent's position ('A') and the goal's position ('G').

## Testing the Environment

Before training an agent, it's a good practice to test the environment to make sure it behaves as expected. We can do this by taking a few random actions.

```python
# In a separate file or after the class definition
if __name__ == '__main__':
    env = GridWorld1D()
    obs, info = env.reset()
    env.render()

    for _ in range(5):
        action = env.action_space.sample()  # Take a random action
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated:
            print("Goal reached!")
            break
```

## The Sparse Reward Problem and Credit Assignment

In our GridWorld, the agent only receives a reward when it reaches the goal. All other actions result in a reward of 0. This is known as a **sparse reward** setting.

Imagine the agent takes a sequence of 20 actions, and only the very last action leads to the goal. How does the agent know which of the preceding 19 actions were good and which were bad? This is the **credit assignment problem**. It's difficult for the agent to figure out which actions in a long sequence were crucial for achieving the final reward.

In our simple 1D case, the solution is obvious to us: always move right. But for an RL agent that learns from scratch, it's not so simple. The agent might move left and right randomly for a long time before accidentally stumbling upon the goal. Only then can it start to learn that moving right is beneficial.

## A Simple Policy

We will use a very simple policy representation: a table that maps each state to a probability distribution over actions. We'll initialize this policy randomly and then update it based on the rewards we receive.

Our policy will be a 2D NumPy array of size `(state_space_size, action_space_size)`. `policy[s, a]` will store the probability of taking action `a` in state `s`.

```python
policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
```

This initializes a uniform random policy, where the agent is equally likely to move left or right in any state.

## Training the Agent

We will use a simple Monte Carlo policy gradient method (also known as REINFORCE). The basic idea is:

1.  Run an episode using the current policy.
2.  If the episode was successful (i.e., the agent reached the goal), increase the probability of the actions taken during that episode.
3.  If the episode was not successful, we don't make any updates (in this simple version).

```python
learning_rate = 0.1
n_episodes = 1000

for episode in range(n_episodes):
    episode_history = []
    obs, info = env.reset()
    terminated = False

    while not terminated:
        action_probs = policy[obs]
        action = np.random.choice(env.action_space.n, p=action_probs)
        
        episode_history.append((obs, action))
        
        obs, reward, terminated, truncated, info = env.step(action)

    # If the goal was reached, update the policy
    if reward == 1:
        for state, action in episode_history:
            policy[state, action] += learning_rate * (1 - policy[state, action])
            # Normalize the probabilities
            policy[state] /= np.sum(policy[state])

print("Training finished.")
print("Learned policy:")
print(policy)
```

### Explanation of the Training Loop

- We run for a fixed number of episodes.
- In each episode, we store the state-action pairs in `episode_history`.
- We select actions based on the current policy probabilities (`np.random.choice`).
- After the episode ends, if the final reward was 1, we iterate through the history and update the policy. The update rule `policy[s, a] += lr * (1 - policy[s, a])` increases the probability of the chosen action `a` in state `s`.
- We then re-normalize the probabilities for that state to ensure they sum to 1.

## Visualizing the Learned Policy

After training, we can visualize the policy to see what the agent has learned. We will create a bar chart showing the probability of moving right for each state.

```python
import matplotlib.pyplot as plt

# Extract the probability of moving right for each state
prob_move_right = policy[:, 1]

plt.figure(figsize=(10, 6))
plt.bar(range(env.observation_space.n), prob_move_right, color='skyblue')
plt.xlabel('State (Position)')
plt.ylabel('Probability of Moving Right')
plt.title('Learned Policy: Probability of Moving Right in Each State')
plt.xticks(range(env.observation_space.n))
plt.grid(axis='y', linestyle='--')
plt.show()
```

If the training was successful, we should see that for most states, the probability of moving right is close to 1.

## Complete Code

The complete, runnable code will be provided in the `gridworld_1d.py` file.

## Dependencies

To run the code, you will need `numpy`, `matplotlib`, and `gymnasium`. You can install them using `uv` with the provided `pyproject.toml` file.

```bash
uv pip install -r requirements.txt
```

## Understanding the Results

When you run the code, you may notice something interesting: the agent achieves a 100% success rate during training, but the learned policy is not always optimal. For example, the agent might learn to prefer moving left in some states, even though always moving right would be the shortest path to the goal.

This happens because of the sparse reward problem. Since the agent only receives feedback at the goal, it cannot distinguish between efficient paths (9 steps) and inefficient paths (15+ steps) as long as both eventually reach the goal. The agent learns that certain actions "work" (they eventually lead to reward), but it doesn't learn which actions are best.

This is a fundamental challenge in RL with sparse rewards:

- **All successful paths receive the same reward**, regardless of efficiency.
- **The agent cannot learn from failures** in our simple implementation, because unsuccessful episodes provide no information.
- **Random exploration** can lead to suboptimal but successful behaviors being reinforced.

To address this, more sophisticated RL algorithms use techniques like:

- **Reward shaping**: Adding intermediate rewards to guide learning (e.g., small rewards for moving closer to the goal).
- **Temporal difference learning**: Methods like Q-learning that can learn from every step, not just successful episodes.
- **Exploration strategies**: Techniques to ensure the agent explores efficiently and discovers better paths.

## Conclusion

In this tutorial, you have learned how to create a custom Gymnasium environment, and you have seen firsthand how sparse rewards can make learning difficult for an RL agent. We implemented a simple policy-based method to solve the 1D GridWorld and visualized the resulting policy. This example highlights the importance of exploration in RL, as the agent must first discover the reward before it can learn to obtain it efficiently.

The key takeaway is that sparse rewards create a **credit assignment problem**: it's hard for the agent to know which actions in a long sequence were truly responsible for the final reward. This motivates the development of more sophisticated RL algorithms that can handle delayed rewards more effectively.
