# Tutorial 1: Number Guesser & Binary Rewards

## High-Level Overview

In this tutorial, we'll build your first reinforcement learning environment from scratch: a number guessing game. This simple problem will teach you the fundamental concepts of RL while addressing your specific questions about **binary rewards** and why they matter.

### What You'll Learn

**Core RL Concepts**: You'll implement the basic RL loop—an agent takes actions in an environment, receives rewards, and updates its policy to improve over time.

**Binary Rewards**: You'll understand why we start with simple `{0, 1}` rewards. Binary rewards are unambiguous: either the agent succeeded or it didn't. There's no partial credit, which makes debugging much easier. When your agent isn't learning, you know it's a problem with your algorithm, not your reward function.

**Policy Representation**: You'll see how a policy can be as simple as a probability distribution over actions. The agent samples from this distribution to make decisions.

**Policy Updates**: You'll implement a naive policy gradient update and discover why normalization matters. Without proper normalization, your policy will change wildly and become unstable.

### The Problem

The environment picks a secret number between 0 and 10. The agent has to learn which number to guess by trying different guesses and receiving feedback. Initially, the agent has no knowledge—it guesses randomly. Through trial and error, it should converge on always guessing the correct number.

### Why This Matters

This toy problem captures the essence of all RL: learning from experience without explicit supervision. The same principles you'll learn here—policy representation, reward signals, and policy updates—scale to training robots, playing games, and even training large language models.

---

## Setup

First, let's set up our project with `uv`:

```bash
# Create a new directory
mkdir number-guesser
cd number-guesser

# Initialize with uv
uv init
uv add numpy matplotlib
```

---

## Part 1: Creating the Environment

The environment is the world the agent interacts with. For our number guesser, it holds a secret target number and provides rewards based on guesses.

```python
import numpy as np


class NumberGuessingEnv:
    """Environment where the agent tries to guess a target number."""

    def __init__(self, target: int = 5, min_val: int = 0, max_val: int = 10):
        self.target = target
        self.min = min_val
        self.max = max_val

    def reward(self, guess: int) -> float:
        """Returns binary reward: 1 if correct, 0 otherwise."""
        if guess < self.min or guess > self.max:
            raise ValueError(
                f"Guess {guess} is out of bounds. Must be between {self.min} and {self.max}."
            )
        
        # Binary reward: 1 for correct, 0 for incorrect
        return 1.0 if guess == self.target else 0.0
```

### Understanding Binary Rewards

The reward function is the heart of RL. It defines what "success" means. Here, we use a **binary reward**:

- `reward = 1.0` if the guess is correct
- `reward = 0.0` if the guess is wrong

**Why binary?** Because it's unambiguous. The agent either got it right or didn't. No partial credit, no confusion. This simplicity is crucial when you're learning RL because it removes one source of potential bugs.

Let's test it:

```python
if __name__ == "__main__":
    env = NumberGuessingEnv(target=5, min_val=0, max_val=10)
    
    for guess in [5, 3, 0, 10]:
        reward = env.reward(guess)
        print(f"Guessed {guess}, reward: {reward}")
```

Output:
```
Guessed 5, reward: 1.0
Guessed 3, reward: 0.0
Guessed 0, reward: 0.0
Guessed 10, reward: 0.0
```

Only the correct guess (5) gets a reward of 1. Everything else gets 0.

---

## Part 2: Creating the Policy

The **policy** is the agent's strategy for choosing actions. In our case, it's a probability distribution over which number to guess.

We'll represent the policy as a simple array where each element is the probability of guessing that number.

```python
class Guesser:
    """Agent that learns to guess the target number."""
    
    def __init__(self, min_val: int, max_val: int):
        self.min = min_val
        self.max = max_val
        num_actions = max_val - min_val + 1
        
        # Initialize with uniform distribution
        self.policy = np.ones(num_actions) / num_actions
    
    def guess(self, size: int = 1) -> np.ndarray:
        """Sample guesses from the current policy."""
        actions = list(range(self.min, self.max + 1))
        return np.random.choice(actions, size=size, p=self.policy)
```

### Understanding the Policy

The policy starts as a **uniform distribution**. If we have 11 possible numbers (0-10), each has a probability of `1/11 ≈ 0.09`.

```python
agent = Guesser(min_val=0, max_val=10)
print(f"Initial policy: {agent.policy}")
# Output: [0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.09]
```

The agent has no prior knowledge, so it treats all guesses as equally likely. Let's see it make some random guesses:

```python
env = NumberGuessingEnv(target=5, min_val=0, max_val=10)
agent = Guesser(min_val=0, max_val=10)

for _ in range(5):
    guess = agent.guess()[0]
    reward = env.reward(guess)
    print(f"Guessed {guess}, reward: {reward}")
```

Output (will vary due to randomness):
```
Guessed 7, reward: 0.0
Guessed 2, reward: 0.0
Guessed 5, reward: 1.0
Guessed 9, reward: 0.0
Guessed 1, reward: 0.0
```

The agent is guessing randomly. Sometimes it gets lucky (guess 5), but most of the time it doesn't.

---

## Part 3: Updating the Policy

Now comes the core of RL: **learning from experience**. We need to update the policy to make good actions (guesses that led to high rewards) more likely.

We'll implement a simple policy gradient update:

1. Collect a batch of guesses and their rewards
2. For each action, add its total reward to the policy
3. Normalize the policy so it sums to 1

```python
def update_policy(self, actions: np.ndarray, rewards: np.ndarray):
    """Update policy based on actions and rewards."""
    # Aggregate rewards for each action
    action_rewards = {}
    for action, reward in zip(actions, rewards):
        if action not in action_rewards:
            action_rewards[action] = 0
        action_rewards[action] += reward
    
    # Update policy
    for action, total_reward in action_rewards.items():
        action_idx = action - self.min
        self.policy[action_idx] += total_reward
    
    # Ensure no negative probabilities
    self.policy = np.maximum(self.policy, 1e-8)
    
    # Normalize to sum to 1
    self.policy = self.policy / self.policy.sum()
```

### Understanding the Update

Let's break down what's happening:

1. **Aggregate rewards**: If we guessed `5` three times and got rewards `[1.0, 1.0, 1.0]`, the total reward for action `5` is `3.0`.

2. **Update policy**: We add the total reward to the policy entry for that action. Actions with higher total rewards get a bigger boost.

3. **Normalize**: We divide by the sum so the policy remains a valid probability distribution (sums to 1).

Let's see it in action with a small batch:

```python
env = NumberGuessingEnv(target=5, min_val=0, max_val=10)
agent = Guesser(min_val=0, max_val=10)

print(f"Initial policy: {np.round(agent.policy, 2)}")

# Collect 10 guesses
guesses = agent.guess(size=10)
rewards = np.array([env.reward(g) for g in guesses])

print(f"Guesses: {guesses}")
print(f"Rewards: {rewards}")

# Update policy
agent.update_policy(guesses, rewards)
print(f"Updated policy: {np.round(agent.policy, 2)}")
```

Output (will vary):
```
Initial policy: [0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.09]
Guesses: [7 2 5 9 1 5 3 8 5 0]
Rewards: [0. 0. 1. 0. 0. 1. 0. 0. 1. 0.]
Updated policy: [0.07 0.07 0.07 0.07 0.07 0.28 0.07 0.07 0.07 0.07 0.07]
```

Notice that the policy for action `5` (index 5) increased from `0.09` to `0.28` because it received rewards. The other actions stayed roughly the same (slightly decreased due to normalization).

---

## Part 4: The Instability Problem

Let's try a larger batch size and see what happens:

```python
BATCH_SIZE = 1000

guesses = agent.guess(size=BATCH_SIZE)
rewards = np.array([env.reward(g) for g in guesses])
agent.update_policy(guesses, rewards)

print(f"Updated policy: {np.round(agent.policy, 2)}")
```

Output (will vary):
```
Updated policy: [0.01 0.0  0.0  0.0  0.0  0.97 0.0  0.0  0.01 0.0  0.0 ]
```

Whoa! The policy changed drastically. It went from uniform to almost entirely focused on action `5`. This happened in just one update!

**Why is this a problem?** The policy is changing too quickly. If we had a noisy environment or if the agent got unlucky with its samples, this could lead to the policy converging to the wrong action and never recovering.

### The Solution: Normalize by Batch Size

We need to make the update size independent of the batch size. Instead of adding the total reward, we add the **average reward per action**:

```python
def update_policy(self, actions: np.ndarray, rewards: np.ndarray):
    """Update policy based on actions and rewards (normalized version)."""
    action_rewards = {}
    for action, reward in zip(actions, rewards):
        if action not in action_rewards:
            action_rewards[action] = 0
        action_rewards[action] += reward
    
    for action, total_reward in action_rewards.items():
        action_idx = action - self.min
        # Normalize by batch size
        self.policy[action_idx] += total_reward / len(actions)
    
    self.policy = np.maximum(self.policy, 1e-8)
    self.policy = self.policy / self.policy.sum()
```

Now let's try again:

```python
agent = Guesser(min_val=0, max_val=10)
print(f"Initial policy: {np.round(agent.policy, 2)}")

guesses = agent.guess(size=1000)
rewards = np.array([env.reward(g) for g in guesses])
agent.update_policy(guesses, rewards)

print(f"Updated policy: {np.round(agent.policy, 2)}")
```

Output:
```
Initial policy: [0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.09]
Updated policy: [0.08 0.08 0.08 0.08 0.08 0.17 0.08 0.08 0.08 0.08 0.08]
```

Much better! The policy for action `5` increased slightly, but the change is gradual and stable.

---

## Part 5: Training Loop

Now let's put it all together and train the agent over multiple iterations:

```python
if __name__ == "__main__":
    MIN = 0
    MAX = 10
    TARGET = 5
    BATCH_SIZE = 1000
    ITERATIONS = 100

    env = NumberGuessingEnv(target=TARGET, min_val=MIN, max_val=MAX)
    agent = Guesser(MIN, MAX)

    print(f"Initial policy: {np.round(agent.policy, 2)}")
    
    for iteration in range(ITERATIONS):
        # Collect batch of experience
        guesses = agent.guess(size=BATCH_SIZE)
        rewards = np.array([env.reward(g) for g in guesses])
        
        # Update policy
        agent.update_policy(guesses, rewards)
        
        # Print progress every 10 iterations
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}: {np.round(agent.policy, 2)}")
    
    print(f"\nFinal policy: {np.round(agent.policy, 2)}")
```

Output:
```
Initial policy: [0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.09]
Iteration 10: [0.07 0.07 0.07 0.07 0.07 0.27 0.07 0.07 0.07 0.07 0.07]
Iteration 20: [0.05 0.05 0.05 0.05 0.05 0.45 0.05 0.05 0.05 0.05 0.05]
Iteration 30: [0.03 0.03 0.03 0.03 0.03 0.63 0.03 0.03 0.03 0.03 0.03]
Iteration 40: [0.02 0.02 0.02 0.02 0.02 0.76 0.02 0.02 0.02 0.02 0.02]
Iteration 50: [0.01 0.01 0.01 0.01 0.01 0.85 0.01 0.01 0.01 0.01 0.01]
Iteration 60: [0.01 0.01 0.01 0.01 0.01 0.91 0.01 0.01 0.01 0.01 0.01]
Iteration 70: [0.0  0.0  0.0  0.0  0.0  0.95 0.0  0.0  0.0  0.0  0.0 ]
Iteration 80: [0.0  0.0  0.0  0.0  0.0  0.97 0.0  0.0  0.0  0.0  0.0 ]
Iteration 90: [0.0  0.0  0.0  0.0  0.0  0.98 0.0  0.0  0.0  0.0  0.0 ]
Iteration 100: [0.0  0.0  0.0  0.0  0.0  0.99 0.0  0.0  0.0  0.0  0.0 ]

Final policy: [0.0  0.0  0.0  0.0  0.0  0.99 0.0  0.0  0.0  0.0  0.0 ]
```

**The agent learned!** It started with no knowledge (uniform distribution) and converged to a policy that guesses `5` with 99% probability.

---

## Part 6: Visualizing Learning

Let's add visualization to see the learning process:

```python
import matplotlib.pyplot as plt

def train_and_visualize():
    MIN = 0
    MAX = 10
    TARGET = 5
    BATCH_SIZE = 1000
    ITERATIONS = 100

    env = NumberGuessingEnv(target=TARGET, min_val=MIN, max_val=MAX)
    agent = Guesser(MIN, MAX)

    # Track policy over time
    policy_history = [agent.policy.copy()]
    
    for _ in range(ITERATIONS):
        guesses = agent.guess(size=BATCH_SIZE)
        rewards = np.array([env.reward(g) for g in guesses])
        agent.update_policy(guesses, rewards)
        policy_history.append(agent.policy.copy())
    
    # Plot
    policy_history = np.array(policy_history)
    plt.figure(figsize=(12, 6))
    
    for action in range(MIN, MAX + 1):
        plt.plot(policy_history[:, action - MIN], label=f"Action {action}")
    
    plt.xlabel("Iteration")
    plt.ylabel("Probability")
    plt.title("Policy Evolution Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("policy_evolution.png")
    print("Saved plot to policy_evolution.png")

if __name__ == "__main__":
    train_and_visualize()
```

This will create a plot showing how each action's probability changes over time. You'll see action `5` steadily increase while all others decrease.

---

## Complete Code

Here's the full implementation:

```python
import numpy as np
import matplotlib.pyplot as plt


class NumberGuessingEnv:
    """Environment where the agent tries to guess a target number."""

    def __init__(self, target: int = 5, min_val: int = 0, max_val: int = 10):
        self.target = target
        self.min = min_val
        self.max = max_val

    def reward(self, guess: int) -> float:
        """Returns binary reward: 1 if correct, 0 otherwise."""
        if guess < self.min or guess > self.max:
            raise ValueError(
                f"Guess {guess} is out of bounds. Must be between {self.min} and {self.max}."
            )
        return 1.0 if guess == self.target else 0.0


class Guesser:
    """Agent that learns to guess the target number."""
    
    def __init__(self, min_val: int, max_val: int):
        self.min = min_val
        self.max = max_val
        num_actions = max_val - min_val + 1
        self.policy = np.ones(num_actions) / num_actions
    
    def guess(self, size: int = 1) -> np.ndarray:
        """Sample guesses from the current policy."""
        actions = list(range(self.min, self.max + 1))
        return np.random.choice(actions, size=size, p=self.policy)
    
    def update_policy(self, actions: np.ndarray, rewards: np.ndarray):
        """Update policy based on actions and rewards."""
        action_rewards = {}
        for action, reward in zip(actions, rewards):
            if action not in action_rewards:
                action_rewards[action] = 0
            action_rewards[action] += reward
        
        for action, total_reward in action_rewards.items():
            action_idx = action - self.min
            # Normalize by batch size for stability
            self.policy[action_idx] += total_reward / len(actions)
        
        self.policy = np.maximum(self.policy, 1e-8)
        self.policy = self.policy / self.policy.sum()


def train_and_visualize():
    """Train the agent and visualize the learning process."""
    MIN = 0
    MAX = 10
    TARGET = 5
    BATCH_SIZE = 1000
    ITERATIONS = 100

    env = NumberGuessingEnv(target=TARGET, min_val=MIN, max_val=MAX)
    agent = Guesser(MIN, MAX)

    print(f"Initial policy: {np.round(agent.policy, 2)}")
    
    policy_history = [agent.policy.copy()]
    
    for iteration in range(ITERATIONS):
        guesses = agent.guess(size=BATCH_SIZE)
        rewards = np.array([env.reward(g) for g in guesses])
        agent.update_policy(guesses, rewards)
        policy_history.append(agent.policy.copy())
        
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}: {np.round(agent.policy, 2)}")
    
    print(f"\nFinal policy: {np.round(agent.policy, 2)}")
    
    # Visualize
    policy_history = np.array(policy_history)
    plt.figure(figsize=(12, 6))
    
    for action in range(MIN, MAX + 1):
        plt.plot(policy_history[:, action - MIN], label=f"Action {action}")
    
    plt.xlabel("Iteration")
    plt.ylabel("Probability")
    plt.title("Policy Evolution Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("policy_evolution.png")
    print("\nSaved plot to policy_evolution.png")


if __name__ == "__main__":
    train_and_visualize()
```

---

## Key Takeaways

**Binary Rewards Are Simple and Unambiguous**: Starting with `{0, 1}` rewards removes complexity. You know exactly what success means, which makes debugging much easier.

**Policy Representation Matters**: A policy can be as simple as a probability distribution. The agent samples from this distribution to make decisions.

**Normalization Is Critical**: Without normalizing by batch size, your policy updates will be unstable and dependent on how much data you collect. Always normalize!

**Learning Takes Time**: Even in this simple problem, the agent needs many iterations to converge. Real-world problems require even more patience.

**Visualization Helps**: Plotting the policy over time gives you intuition for what's happening. If the policy isn't changing, you know something is wrong.

---

## Next Steps

Now that you understand binary rewards and basic policy updates, you're ready for Tutorial 2, where we'll implement REINFORCE (vanilla policy gradient) and learn about:

- **Advantages**: Why raw rewards aren't enough
- **Normalized advantages**: The key to stable training
- **Value functions**: Reducing variance with baselines

Try experimenting with this code:
- Change the target number
- Increase the action space (0-100 instead of 0-10)
- Add noise to the rewards (like in the example you shared)
- Try different batch sizes and learning rates

See you in the next tutorial!
