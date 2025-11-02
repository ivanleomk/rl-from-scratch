"""
Tutorial 1: Number Guesser with Binary Rewards

A simple RL environment where an agent learns to guess a target number.
Demonstrates binary rewards, policy representation, and policy updates.
"""

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
        
        # Initialize with uniform distribution
        self.policy = np.ones(num_actions) / num_actions
    
    def guess(self, size: int = 1) -> np.ndarray:
        """Sample guesses from the current policy."""
        actions = list(range(self.min, self.max + 1))
        return np.random.choice(actions, size=size, p=self.policy)
    
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
            # Normalize by batch size for stability
            self.policy[action_idx] += total_reward / len(actions)
        
        # Ensure no negative probabilities
        self.policy = np.maximum(self.policy, 1e-8)
        
        # Normalize to sum to 1
        self.policy = self.policy / self.policy.sum()


def train_and_visualize():
    """Train the agent and visualize the learning process."""
    # Hyperparameters
    MIN = 0
    MAX = 10
    TARGET = 5
    BATCH_SIZE = 1000
    ITERATIONS = 100

    # Initialize environment and agent
    env = NumberGuessingEnv(target=TARGET, min_val=MIN, max_val=MAX)
    agent = Guesser(MIN, MAX)

    print(f"Target number: {TARGET}")
    print(f"Initial policy: {np.round(agent.policy, 2)}\n")
    
    # Track policy over time
    policy_history = [agent.policy.copy()]
    
    # Training loop
    for iteration in range(ITERATIONS):
        # Collect batch of experience
        guesses = agent.guess(size=BATCH_SIZE)
        rewards = np.array([env.reward(g) for g in guesses])
        
        # Update policy
        agent.update_policy(guesses, rewards)
        policy_history.append(agent.policy.copy())
        
        # Print progress
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1:3d}: {np.round(agent.policy, 2)}")
    
    print(f"\nFinal policy: {np.round(agent.policy, 2)}")
    print(f"Agent now guesses {TARGET} with {agent.policy[TARGET - MIN]:.1%} probability")
    
    # Visualize policy evolution
    policy_history = np.array(policy_history)
    plt.figure(figsize=(12, 6))
    
    for action in range(MIN, MAX + 1):
        label = f"Action {action}"
        if action == TARGET:
            label += " (target)"
        plt.plot(policy_history[:, action - MIN], label=label, linewidth=2 if action == TARGET else 1)
    
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.title("Policy Evolution Over Time", fontsize=14)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("policy_evolution.png", dpi=150)
    print("\nSaved plot to policy_evolution.png")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    train_and_visualize()
