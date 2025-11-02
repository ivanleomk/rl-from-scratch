# Tutorial 3: Building a CartPole Environment Wrapper

Welcome to the third tutorial in our Reinforcement Learning (RL) curriculum! In this hands-on guide, you will learn how to create a custom wrapper for a standard RL environment from the [Gymnasium](https://gymnasium.farama.org/) library. Building wrappers is a fundamental skill for any RL practitioner, as it allows you to modify and extend existing environments to suit your specific research or application needs.

By the end of this tutorial, you will understand:
- How to use existing Gymnasium environments like `CartPole-v1`.
- The concept of continuous state spaces and discrete action spaces.
- How to create a Python class to wrap an environment, providing a clean interface for interaction.
- How to track and manage episode statistics like rewards and step counts.
- The process of training a simple agent and visualizing its performance.

---

## The CartPole Problem

The CartPole problem is a classic RL benchmark. The goal is to balance a pole that is attached by an un-actuated joint to a cart. The cart can move along a frictionless track, and the only control you have is to push the cart either to the left or to the right.

> A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart. [1]

### State and Action Spaces

To solve this, the agent needs to understand the state of the environment and decide which action to take. 

- **Continuous State Space**: The state is represented by four continuous numbers: 
  1. Cart Position
  2. Cart Velocity
  3. Pole Angle
  4. Pole Angular Velocity

- **Discrete Action Space**: The agent can take one of two possible discrete actions:
  - `0`: Push the cart to the left.
  - `1`: Push the cart to the right.

An episode ends if the pole falls over (exceeds a certain angle) or the cart moves too far from the center.

---

## Part 1: The Environment Wrapper

While we could use the Gymnasium environment directly, creating a wrapper class provides several benefits. It helps abstract away the low-level details of the environment, allows us to add custom logic for tracking statistics, and provides a clean, reusable interface for our RL agents.

We will create a `CartPoleWrapper` class that handles loading the environment, resetting it for new episodes, executing steps, and rendering the simulation.

Here is the basic structure in `cartpole_wrapper.py`:

```python
import gymnasium as gym
import numpy as np

class CartPoleWrapper:
    def __init__(self, render_mode=None):
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.render_mode = render_mode
        
        self.total_reward = 0
        self.total_steps = 0
        self.episode_history = []

    def reset(self):
        # Reset the environment and statistics
        initial_observation, info = self.env.reset()
        if self.total_steps > 0:
            self.episode_history.append({
                "reward": self.total_reward,
                "steps": self.total_steps
            })
        self.total_reward = 0
        self.total_steps = 0
        return initial_observation

    def step(self, action):
        # Take a step, update stats, and handle rendering
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += reward
        self.total_steps += 1
        if self.render_mode:
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode:
            self.env.render()

    def close(self):
        self.env.close()
```

### Explanation
- **`__init__(self, render_mode=None)`**: The constructor initializes the `CartPole-v1` environment. The `render_mode` argument allows us to specify if we want to see the graphical output (`"human"`). We also initialize variables to track the reward and steps for the current episode and a list to store the history of all episodes.
- **`reset(self)`**: This method is called at the beginning of each new episode. It resets the underlying Gymnasium environment and also archives the stats from the completed episode before resetting the counters.
- **`step(self, action)`**: This method executes the chosen `action` in the environment. It updates our `total_reward` and `total_steps` counters and returns the standard `(observation, reward, terminated, truncated, info)` tuple that RL agents expect.

---

## Part 2: Testing with a Random Policy

Now that we have our wrapper, let's test it by running a few episodes with a completely random policy. This is a good sanity check to ensure our wrapper works as expected and to establish a performance baseline.

```python
# In cartpole_wrapper.py

def run_random_policy(episodes=10):
    env = CartPoleWrapper(render_mode="human")
    for episode in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action = env.env.action_space.sample() # Choose a random action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    env.close()
    print("Finished running random policy.")
    print("Episode history:", env.episode_history)

if __name__ == "__main__":
    run_random_policy()
```

When you run this script, you will see the CartPole environment for 10 episodes. The cart will move randomly, and the pole will likely fall quickly. The output will show the total reward for each episode, which will be quite low.

---

## Part 3: Training a Simple Q-Learning Agent

To demonstrate how to use our wrapper for training, we will implement a simple Q-learning agent. Since the state space is continuous, we need to discretize it to use a standard Q-table. This is a simplification, but it serves to illustrate the training process.

We will add the agent and training loop to our `cartpole_wrapper.py` file.

```python
# (Add to cartpole_wrapper.py)

class QLearningAgent:
    def __init__(self, action_space, observation_space, bins=[10, 10, 10, 10], lr=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Discretize the continuous observation space
        self.state_bins = [
            np.linspace(-2.4, 2.4, bins[0]),
            np.linspace(-4, 4, bins[1]),
            np.linspace(-0.2095, 0.2095, bins[2]),
            np.linspace(-4, 4, bins[3]),
        ]
        self.q_table = np.zeros(tuple(bins) + (action_space.n,))

    def discretize_state(self, state):
        indices = []
        for i, val in enumerate(state):
            indices.append(np.digitize(val, self.state_bins[i]) - 1)
        return tuple(indices)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        discrete_state = self.discretize_state(state)
        return np.argmax(self.q_table[discrete_state])

    def update_q_table(self, state, action, reward, next_state, done):
        discrete_state = self.discretize_state(state)
        next_discrete_state = self.discretize_state(next_state)
        
        old_value = self.q_table[discrete_state + (action,)]
        next_max = np.max(self.q_table[next_discrete_state])
        
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[discrete_state + (action,)] = new_value

        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

---

## Part 4: Training and Visualization

Finally, let's write the training loop and add code to visualize the results using `matplotlib`.

```python
# (Add to cartpole_wrapper.py)
import matplotlib.pyplot as plt

def train_agent(episodes=5000):
    env = CartPoleWrapper()
    agent = QLearningAgent(env.env.action_space, env.env.observation_space)
    
    for episode in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if done:
                reward = -10 # Penalize failure

            agent.update_q_table(obs, action, reward, next_obs, done)
            obs = next_obs
        
        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}/{episodes}, Epsilon: {agent.epsilon:.4f}")

    env.close()
    return env.episode_history

def plot_rewards(episode_history):
    rewards = [e["reward"] for e in episode_history]
    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    plt.title("Episode Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig("episode_rewards.png")
    plt.show()

if __name__ == "__main__":
    # run_random_policy()
    history = train_agent()
    plot_rewards(history)
```

After running the training, an `episode_rewards.png` image will be generated. You should see a clear upward trend in the rewards as the agent learns to balance the pole for longer durations, a significant improvement over the random policy.

![Example Training Plot](https://private-us-east-1.manuscdn.com/sessionFile/R9UCUmckgRhYrj6TDQmmHX/sandbox/gEr3gcEsD9Uvyu8Lp5ZPvY-images_1762098007713_na1fn_L2hvbWUvdWJ1bnR1L2NhcnRwb2xlX3R1dG9yaWFsL2V4YW1wbGVfdHJhaW5pbmdfcGxvdA.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvUjlVQ1VtY2tnUmhZcmo2VERRbW1IWC9zYW5kYm94L2dFcjNnY0VzRDlVdnl1OExwNVpQdlktaW1hZ2VzXzE3NjIwOTgwMDc3MTNfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyTmhjblJ3YjJ4bFgzUjFkRzl5YVdGc0wyVjRZVzF3YkdWZmRISmhhVzVwYm1kZmNHeHZkQS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=MkcuAFXyAs3wqrIekn7uEfouKTadpp7f6vy-dV0lUEjOne640PTYbDZqBSOBgCLgxYINmeYgTxvQVMgrK6KNqdjXg1gFH6OgVau-YHattBDdd8JBEBfdNDXzaraXPu-b36QDMBLfkgvYxd~Y~XzUH4l47UVYAP8U~PAFdNeg~2PDh-r~iY0-CmDPpLyYHtX7EDX2FmaVptwe1XoL94A~b2ERCBuFHR1TQVOyj~8kGlF~EEuE2yeHSDZ2FuRGtzIVdJdkg2BipixiAjQ7Q-Fcjtl0K5MhVyZ1bqk4PI7ufuCL9V8gPghs-ctXS36J5vkTXlHI~NbntBI5t2NgcjAMMQ__)

The plot above shows an example training run over 100 episodes. The light blue line represents individual episode rewards (which can be quite noisy), while the orange line shows the moving average, clearly indicating the learning progress.

## Project Structure

Your final project directory should look like this:

```
/cartpole_tutorial
|-- pyproject.toml
|-- cartpole_wrapper.py
|-- README.md
```

- **`pyproject.toml`**: Defines the project dependencies (`numpy`, `matplotlib`, `gymnasium`).
- **`cartpole_wrapper.py`**: Contains the `CartPoleWrapper`, the `QLearningAgent`, and the main training logic.
- **`README.md`**: This tutorial file.

## Conclusion

Congratulations! You have successfully built a wrapper for a standard RL environment, trained a simple agent, and visualized its learning progress. This wrapper-based approach is highly scalable and is used in professional RL projects to manage complex environments and experiments.

## References
[1] Barto, A. G., Sutton, R. S., & Anderson, C. W. (1983). Neuronlike adaptive elements that can solve difficult learning control problems. *IEEE Transactions on Systems, Man, and Cybernetics*, (5), 834–846.
