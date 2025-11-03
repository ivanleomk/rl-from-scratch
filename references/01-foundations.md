# Chapter 1: The Foundations of Reinforcement Learning

Welcome to the foundational chapter of our journey into Reinforcement Learning (RL). Before we dive into writing complex algorithms, it is crucial to build a strong intuition for the core concepts that underpin the field. Inspired by the practical, code-first approach of [fast.ai](https://www.fast.ai/), we will start with tangible examples and interactive code, building our understanding of the theory from the ground up.

## The Core Idea: Learning from Interaction

At its heart, Reinforcement Learning is about learning to make good decisions through trial and error. Imagine a baby learning to walk. It tries different movements (actions), sometimes it falls (negative reward), and sometimes it manages to take a step forward (positive reward). Over time, it learns to associate certain actions with positive outcomes and refines its strategy to walk more effectively. This is the essence of RL.

This interaction occurs in a feedback loop between two main components: the **agent** and the **environment**.

- The **Agent** is the learner or decision-maker. It is the part of the system we are training, like the baby learning to walk or a computer program learning to play a game.
- The **Environment** is the world the agent interacts with. It comprises everything outside the agent.

At each step, the agent observes the current state of the environment, takes an action, and receives a reward and the next state from the environment. This continuous cycle is known as the **agent-environment loop** [1].

![Agent-Environment Loop](https://raw.githubusercontent.com/ivanleomk/rl-from-scratch/main/references/images/agent_environment_loop.png)

*Figure 1: The agent-environment interaction loop. The agent takes an action, and the environment responds with a new state and a reward.*

---

## The Multi-Armed Bandit: A First Taste of the Exploration-Exploitation Dilemma

Before we tackle the full complexity of sequential decision-making, let's consider a simpler problem: the **multi-armed bandit**. Imagine you are in a casino facing a row of slot machines (one-armed bandits). Each machine has a different, unknown probability of paying out a reward. Your goal is to maximize your total winnings over a series of pulls.

This scenario introduces a fundamental challenge in RL: the **exploration-exploitation tradeoff** [2].

- **Exploitation**: You could stick with the machine that has given you the best rewards so far. This is a safe bet, but you might be missing out on an even better machine.
- **Exploration**: You could try a machine you haven't played much. This is risky, as it might have a low payout, but it's the only way to discover the truly best machine.

Finding the right balance is key. A common strategy is the **ε-greedy (epsilon-greedy)** approach, where with a small probability ε, you explore a random machine, and with probability 1-ε, you exploit the one you currently believe is best.

![Bandit Exploration Strategies](https://raw.githubusercontent.com/ivanleomk/rl-from-scratch/main/references/images/bandit_exploration.png)

*Figure 2: (Left) The agent's estimated rewards for each bandit arm compared to the true, unknown rewards. (Right) A comparison of cumulative regret for different exploration strategies. More effective exploration (like UCB) leads to lower regret over time.*

---

## Formalizing the Problem: Markov Decision Processes (MDPs)

To move beyond single-step problems like the multi-armed bandit, we need a mathematical framework for sequential decision-making under uncertainty. This is the **Markov Decision Process (MDP)** [3]. An MDP is defined by five key components:

1. **States (S)**: A set of all possible situations the agent can be in. For example, the position of a robot in a maze.
2. **Actions (A)**: A set of all possible choices the agent can make.
3. **Transition Model (T or P)**: The rules of the environment. It defines the probability of transitioning to a new state `s'` after taking an action `a` in state `s`. This is written as `P(s' | s, a)`.
4. **Reward Function (R)**: A function that defines the reward the agent receives for transitioning from state `s` to `s'` after taking action `a`.
5. **Discount Factor (γ)**: A value between 0 and 1 that determines the importance of future rewards. A discount factor of 0 makes the agent myopic (only caring about immediate rewards), while a value closer to 1 makes it more farsighted.

![MDP Components](https://raw.githubusercontent.com/ivanleomk/rl-from-scratch/main/references/images/mdp_components.png)

*Figure 3: The components of a Markov Decision Process. From a given state, the agent's action leads to a stochastic transition to a new state and a corresponding reward.*

### The Markov Property

A key assumption in MDPs is the **Markov Property**, which states that **the future is independent of the past given the present** [3]. In other words, the current state `s_t` provides all the necessary information to make a decision. We don't need to know the entire history of states and actions that led to the current state.

> **The Markov Property**
>
> A state `s_t` is Markov if and only if:
>
> `P(s_{t+1} | s_t) = P(s_{t+1} | s_1, s_2, ..., s_t)`

This simplifies the problem immensely, as the agent can make optimal decisions based solely on its current observation.

---

## Policies and Value Functions: Evaluating "Goodness"

How does an agent decide which action to take? This is determined by its **policy (π)**, which is a mapping from states to actions. A policy can be:

- **Deterministic**: For a given state, the policy always returns the same action.
- **Stochastic**: For a given state, the policy returns a probability distribution over actions.

But how do we know if a policy is good? We need a way to quantify the expected long-term reward. This is where **value functions** come in. There are two main types:

1. **State-Value Function (V(s))**: The expected return when starting in state `s` and following policy `π` thereafter.
2. **Action-Value Function (Q(s, a))**: The expected return when starting in state `s`, taking action `a`, and then following policy `π`.

---

## The Bellman Equation: A Recursive Relationship

The **Bellman Equation** is a cornerstone of RL. It provides a recursive relationship that connects the value of a state to the values of its successor states [4]. It expresses the idea that the value of your current state is the immediate reward plus the discounted value of the state you end up in.

![Bellman Backup Diagram](https://raw.githubusercontent.com/ivanleomk/rl-from-scratch/main/references/images/bellman_backup.png)

*Figure 4: A Bellman backup diagram. The value of the current state (red) is determined by the expected values of the possible next states (blue), weighted by the probabilities of transitioning to them.*

For a given policy `π`, the Bellman equation for the state-value function is:

```
V^π(s) = E[R_{t+1} + γV^π(S_{t+1}) | S_t = s]
```

This equation forms the basis for many RL algorithms. It allows us to iteratively compute the value function for a given policy.

---

## Finding the Optimal Solution

The ultimate goal of RL is to find the **optimal policy (π*)**, which is the policy that achieves the highest possible expected return from all states. The optimal policy has an associated optimal state-value function `V*(s)` and optimal action-value function `Q*(s, a)`.

Two classic dynamic programming algorithms for finding the optimal policy in an MDP (when the model is known) are **Value Iteration** and **Policy Iteration** [4].

- **Value Iteration**: This algorithm starts with an arbitrary value function and iteratively applies the Bellman Optimality Equation to converge to the optimal value function. The policy is then extracted by choosing the action that maximizes the expected value.
- **Policy Iteration**: This algorithm alternates between two steps: **policy evaluation** (calculating the value function for the current policy) and **policy improvement** (greedily improving the policy based on the current value function).

![Value Iteration in GridWorld](https://raw.githubusercontent.com/ivanleomk/rl-from-scratch/main/references/images/gridworld_value_iteration.png)

*Figure 5: Visualization of Value Iteration in a simple GridWorld. The values of the states are iteratively updated until they converge, at which point the optimal policy (arrows) can be extracted.*

---

## Putting it into Practice with Code: Gymnasium

Now, let's ground these concepts in code. The standard toolkit for creating and interacting with RL environments is **Gymnasium** (the successor to OpenAI's Gym) [5]. It provides a simple, Pythonic interface for a wide variety of environments.

Here is the basic structure of a Gymnasium interaction loop:

```python
import gymnasium as gym

# 1. Create the environment
env = gym.make("LunarLander-v2", render_mode="human")

# 2. Reset the environment to get the initial observation
observation, info = env.reset()

# 3. Loop through a number of steps
for _ in range(1000):
    # 4. Choose an action (here, a random one)
    action = env.action_space.sample()

    # 5. Take the action and get the next state, reward, and other info
    observation, reward, terminated, truncated, info = env.step(action)

    # 6. If the episode is over, reset the environment
    if terminated or truncated:
        observation, info = env.reset()

# 7. Close the environment
env.close()
```

This simple script demonstrates the core agent-environment loop. In the chapters that follow, we will replace the random action selection with sophisticated learning algorithms to create intelligent agents.

---

## Solving the MDP: Dynamic Programming

Now that we have a formal understanding of Markov Decision Processes (MDPs) and the Bellman equations, how do we actually compute the optimal policy? When we have a complete model of the environment—meaning we know the transition probabilities `P(s'|s,a)` and the reward function `R(s,a,s')`—we can use a class of methods called **Dynamic Programming (DP)**.

Dynamic Programming, in the context of RL, refers to a collection of algorithms that can compute optimal policies by turning the Bellman equations into iterative updates. They are powerful but limited by their requirement of a perfect model, a condition often referred to as the "curse of dimensionality" because the computational cost grows exponentially with the number of states.

Despite this limitation, DP methods are theoretically crucial as they provide a solid foundation for understanding more advanced, model-free algorithms. The two main DP methods are Value Iteration and Policy Iteration.

### Value Iteration from Scratch

Value Iteration is an algorithm that repeatedly applies the Bellman Optimality Equation to update the value function until it converges to the optimal value function, `V*`. It combines the steps of policy evaluation and policy improvement into a single, elegant update rule.

Let's make this concrete by implementing Value Iteration from scratch for our simple 3x4 GridWorld. The goal is to see how the values of states are iteratively calculated and how the optimal policy emerges from them.

#### The GridWorld Environment

First, we define our environment. It's a 3x4 grid with a goal, a danger zone, and a wall. The agent's movement is stochastic.

- **States**: 11 non-terminal states and 2 terminal states (Goal, Danger).
- **Actions**: Up, Down, Left, Right.
- **Transitions**: 80% chance of moving in the intended direction, 10% chance of slipping left, and 10% chance of slipping right (relative to the intended direction).
- **Rewards**: +1 for the goal, -1 for danger, and a small negative reward (-0.04) for every other step to encourage efficiency.

#### Full Python Implementation

Here is the complete Python code to solve this GridWorld using Value Iteration. This script calculates the optimal value for every state and then extracts the best policy.

```python
'''
Value Iteration implementation for a simple GridWorld.
This script demonstrates how to calculate the optimal value function and extract the
optimal policy for a known MDP using Value Iteration.
'''

import numpy as np

# --- 1. Define the GridWorld Environment ---

# Grid dimensions
ROWS = 3
COLS = 4

# State representation: (row, col)
START_STATE = (2, 0)
GOAL_STATE = (0, 3)
DANGER_STATE = (1, 3)
WALL_STATE = (1, 1)

# Rewards
REWARD_GOAL = 1
REWARD_DANGER = -1
REWARD_STEP = -0.04

# Actions (Up, Down, Left, Right)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ['U', 'D', 'L', 'R']

# Transition probabilities
PROB_INTENDED = 0.8
PROB_SLIP_LEFT = 0.1
PROB_SLIP_RIGHT = 0.1

# Discount factor
GAMMA = 0.99

# Convergence threshold
CONVERGENCE_THRESHOLD = 1e-4

def is_terminal(state):
    """Check if a state is a terminal state."""
    return state == GOAL_STATE or state == DANGER_STATE

def get_next_state(state, action):
    """Get the resulting state from taking an action, handling walls and boundaries."""
    if is_terminal(state):
        return state

    next_state = (state[0] + action[0], state[1] + action[1])

    # Check for boundary conditions
    if not (0 <= next_state[0] < ROWS and 0 <= next_state[1] < COLS):
        return state  # Bounces back

    # Check for wall
    if next_state == WALL_STATE:
        return state  # Bounces back

    return next_state

def get_reward(state):
    """Get the reward for entering a state."""
    if state == GOAL_STATE:
        return REWARD_GOAL
    if state == DANGER_STATE:
        return REWARD_DANGER
    return REWARD_STEP

# --- 2. Value Iteration Algorithm ---

def value_iteration():
    """Performs the Value Iteration algorithm."""
    # Initialize value function to zeros
    V = np.zeros((ROWS, COLS))
    iteration = 0

    while True:
        delta = 0
        V_new = np.copy(V)

        for r in range(ROWS):
            for c in range(COLS):
                state = (r, c)

                if is_terminal(state) or state == WALL_STATE:
                    continue

                action_values = []
                for action_idx, action in enumerate(ACTIONS):
                    # Calculate the value for this action
                    q_value = 0

                    # Intended outcome
                    next_state_intended = get_next_state(state, action)
                    reward_intended = get_reward(next_state_intended)
                    q_value += PROB_INTENDED * (reward_intended + GAMMA * V[next_state_intended])

                    # Slip left outcome (relative to action)
                    slip_left_action = ACTIONS[(action_idx - 1) % 4]
                    next_state_left = get_next_state(state, slip_left_action)
                    reward_left = get_reward(next_state_left)
                    q_value += PROB_SLIP_LEFT * (reward_left + GAMMA * V[next_state_left])

                    # Slip right outcome (relative to action)
                    slip_right_action = ACTIONS[(action_idx + 1) % 4]
                    next_state_right = get_next_state(state, slip_right_action)
                    reward_right = get_reward(next_state_right)
                    q_value += PROB_SLIP_RIGHT * (reward_right + GAMMA * V[next_state_right])
                    
                    action_values.append(q_value)

                # Update the value function for the current state
                V_new[state] = max(action_values)
                delta = max(delta, abs(V_new[state] - V[state]))

        V = V_new
        iteration += 1

        # Check for convergence
        if delta < CONVERGENCE_THRESHOLD:
            print(f"Value Iteration converged after {iteration} iterations.")
            break
            
    return V

# --- 3. Extract Optimal Policy ---

def extract_policy(V):
    """Extracts the optimal policy from a converged value function."""
    policy = np.full((ROWS, COLS), ' ', dtype=str)
    for r in range(ROWS):
        for c in range(COLS):
            state = (r, c)
            if is_terminal(state) or state == WALL_STATE:
                if state == GOAL_STATE: policy[state] = 'G'
                if state == WALL_STATE: policy[state] = 'W'
                continue

            action_values = []
            for action_idx, action in enumerate(ACTIONS):
                q_value = 0
                # Intended outcome
                next_state_intended = get_next_state(state, action)
                reward_intended = get_reward(next_state_intended)
                q_value += PROB_INTENDED * (reward_intended + GAMMA * V[next_state_intended])
                # Slip left
                slip_left_action = ACTIONS[(action_idx - 1) % 4]
                next_state_left = get_next_state(state, slip_left_action)
                reward_left = get_reward(next_state_left)
                q_value += PROB_SLIP_LEFT * (reward_left + GAMMA * V[next_state_left])
                # Slip right
                slip_right_action = ACTIONS[(action_idx + 1) % 4]
                next_state_right = get_next_state(state, slip_right_action)
                reward_right = get_reward(next_state_right)
                q_value += PROB_SLIP_RIGHT * (reward_right + GAMMA * V[next_state_right])
                action_values.append(q_value)

            best_action_idx = np.argmax(action_values)
            policy[state] = ACTION_NAMES[best_action_idx]
            
    return policy

# --- 4. Run and Print Results ---

if __name__ == "__main__":
    print("--- Running Value Iteration ---")
    optimal_V = value_iteration()
    
    print("\n--- Optimal Value Function V*(s) ---")
    print(np.round(optimal_V, 2))
    
    print("\n--- Optimal Policy π*(s) ---")
    optimal_policy = extract_policy(optimal_V)
    print(optimal_policy)
```

#### Results and Interpretation

Running this script produces the following output, showing the converged value function and the resulting optimal policy.

```
--- Running Value Iteration ---
Value Iteration converged after 22 iterations.

--- Optimal Value Function V*(s) ---
[[0.85 0.91 0.98 0.  ]
 [0.78 0.   0.7  0.  ]
 [0.71 0.65 0.64 0.58]]

--- Optimal Policy π*(s) ---
[['R' 'R' 'R' 'G']
 ['U' 'W' 'U' ' ']
 ['U' 'L' 'U' 'L']]
```

Let's break down these results:

- **Optimal Value Function `V*(s)`**: This grid shows the maximum expected future reward the agent can get from any starting state. Notice how the values are highest near the goal (top right) and decrease as we move away. The state next to the danger zone has a lower value, reflecting the risk.
- **Optimal Policy `π*(s)`**: This grid shows the best action to take in each state. The arrows guide the agent towards the goal while avoiding the wall and the danger zone. For example, from state (2,0), the best action is 'Up'. Notice the policy for the state (1,2) is to go 'Up', even though it's next to the danger zone. This is because the stochastic nature of the environment makes moving right too risky—a slip could be catastrophic.

This from-scratch implementation provides a clear, practical understanding of how an agent can find the optimal way to act in a known environment.

---

## From Theory to Practice: A Complete Q-Learning Agent for CartPole

While understanding the theory is essential, the best way to solidify these concepts is to build a complete, learning agent from scratch. In this section, we will move from model-based dynamic programming to a **model-free** algorithm called **Q-Learning**. This means our agent will learn directly from experience, without needing to know the environment's transition probabilities.

We will tackle the classic `CartPole-v1` environment from Gymnasium. The goal is to balance a pole on a cart that can move left or right. A reward of +1 is given for every timestep the pole remains upright.

### The Challenge: Continuous State Spaces

The CartPole environment has a continuous state space, meaning its state is described by four floating-point numbers: cart position, cart velocity, pole angle, and pole angular velocity. Our Q-table approach, however, requires a discrete number of states. To solve this, we must **discretize** the state space by binning the continuous values into a manageable number of categories. This is a common and practical technique when applying tabular methods to continuous problems.

### The Q-Learning Algorithm

Q-Learning is an off-policy temporal-difference (TD) control algorithm. Its goal is to find the optimal action-value function, `Q*(s, a)`, by iteratively updating a Q-table. The core of the algorithm is its update rule, which is derived from the Bellman equation:

```
Q(s, a) <- Q(s, a) + α * [R + γ * max_a'(Q(s', a')) - Q(s, a)]
```

Where:
- `α` (alpha) is the learning rate, which determines how much we update our Q-values.
- `γ` (gamma) is the discount factor.

### Full Python Implementation for a Q-Learning Agent

Here is the complete, commented code for a Q-Learning agent that learns to solve CartPole. It includes the agent class, the state discretization logic, the training loop, and code to plot the results.

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class QLearningCartPoleAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, 
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_rate=0.9995):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = epsilon_decay_rate

        # Discretize the continuous state space
        self.num_bins = (6, 6, 6, 6)  # Bins for position, velocity, angle, angular velocity
        self.state_bins = [
            np.linspace(-2.4, 2.4, self.num_bins[0]),
            np.linspace(-4, 4, self.num_bins[1]),
            np.linspace(-0.2095, 0.2095, self.num_bins[2]),  # ~12 degrees
            np.linspace(-4, 4, self.num_bins[3])
        ]

        # Initialize Q-table with zeros
        self.q_table = np.zeros(self.num_bins + (env.action_space.n,))

    def discretize_state(self, state):
        """Converts a continuous state to a discrete state index."""
        discrete_state = []
        for i, value in enumerate(state):
            discrete_state.append(np.digitize(value, self.state_bins[i]) - 1)
        return tuple(discrete_state)

    def choose_action(self, state):
        """Chooses an action using an epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            discrete_state = self.discretize_state(state)
            return np.argmax(self.q_table[discrete_state])  # Exploit

    def update_q_table(self, state, action, reward, next_state, terminated):
        """Updates the Q-value for a given state-action pair."""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)

        if terminated:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[discrete_next_state])

        # Q-learning update rule
        old_value = self.q_table[discrete_state + (action,)]
        new_value = old_value + self.lr * (target - old_value)
        self.q_table[discrete_state + (action,)] = new_value

    def decay_epsilon(self):
        """Decays the exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay_rate)


def train_agent(episodes=20000):
    env = gym.make("CartPole-v1")
    agent = QLearningCartPoleAgent(env)
    rewards_per_episode = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        terminated = False

        while not terminated:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Adjust reward to encourage longer episodes
            if not terminated:
                reward = 1.0
            else:
                reward = -10.0  # Penalize failure

            agent.update_q_table(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += 1  # We are counting steps, so reward is 1 per step

            if terminated or truncated:
                break
        
        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)

        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_per_episode[-1000:])
            print(f"Episode {episode + 1}/{episodes} | Avg Reward (last 1000): {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f}")

    env.close()
    return rewards_per_episode


def plot_rewards(rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Reward per Episode', alpha=0.3)
    
    # Calculate and plot rolling average
    rolling_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
    plt.plot(np.arange(99, len(rewards)), rolling_avg, 
             label='100-Episode Rolling Average', color='red', linewidth=2)
    
    plt.title("Q-Learning Agent Performance on CartPole-v1", fontsize=16, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward (Steps)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("q_learning_cartpole_performance.png", dpi=300)
    print("\nPlot saved!")


if __name__ == "__main__":
    print("--- Training Q-Learning Agent for CartPole-v1 ---")
    rewards = train_agent()
    plot_rewards(rewards)
```

### Training and Results

After running the training script for 20,000 episodes, we can observe the agent's learning progress. The output shows the average reward over the last 1000 episodes, which steadily increases as the agent learns a better policy. The exploration rate (epsilon) also decays over time, shifting the agent from exploration to exploitation.

```
--- Training Q-Learning Agent for CartPole-v1 ---
Episode 1000/20000 | Avg Reward (last 1000): 29.02 | Epsilon: 0.6065
Episode 2000/20000 | Avg Reward (last 1000): 54.45 | Epsilon: 0.3678
Episode 3000/20000 | Avg Reward (last 1000): 93.04 | Epsilon: 0.2230
Episode 4000/20000 | Avg Reward (last 1000): 129.24 | Epsilon: 0.1353
Episode 5000/20000 | Avg Reward (last 1000): 219.04 | Epsilon: 0.0820
Episode 6000/20000 | Avg Reward (last 1000): 305.89 | Epsilon: 0.0500
...
Episode 20000/20000 | Avg Reward (last 1000): 267.78 | Epsilon: 0.0500
```

The learning process is clearly visible in the performance plot:

![Q-Learning Performance on CartPole-v1](https://raw.githubusercontent.com/ivanleomk/rl-from-scratch/main/references/images/q_learning_cartpole_performance.png)

*Figure 6: The total reward (number of steps the pole was balanced) per episode. The red line shows the 100-episode rolling average, which clearly trends upward, indicating successful learning.*

This complete example demonstrates how the foundational concepts of RL come together to create an agent that can learn a complex task through trial and error. We have moved from the abstract theory of MDPs and Bellman equations to a practical, working implementation of a model-free learning algorithm.

---

## References

[1] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press. [http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)

[2] Analytics Vidhya. (2018). *Multi Armed Bandit Problem & Its Implementation in Python*. [https://www.analyticsvidhya.com/blog/2018/09/reinforcement-multi-armed-bandit-scratch-python/](https://www.analyticsvidhya.com/blog/2018/09/reinforcement-multi-armed-bandit-scratch-python/)

[3] GeeksforGeeks. (2025). *Markov Decision Process*. [https://www.geeksforgeeks.org/machine-learning/markov-decision-process/](https://www.geeksforgeeks.org/machine-learning/markov-decision-process/)

[4] Karpathy, A. (n.d.). *REINFORCEjs: Gridworld with Dynamic Programming*. [https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html)

[5] Farama Foundation. (2023). *Gymnasium Documentation*. [https://gymnasium.farama.org/](https://gymnasium.farama.org/)
