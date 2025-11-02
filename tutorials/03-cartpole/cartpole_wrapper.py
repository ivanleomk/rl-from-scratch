"""
CartPole Environment Wrapper for Reinforcement Learning
========================================================

This module provides a wrapper class for the Gymnasium CartPole-v1 environment,
along with a simple Q-learning agent for training.

Author: Manus AI
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class CartPoleWrapper:
    """
    A wrapper for the CartPole-v1 environment that provides a clean interface
    and tracks episode statistics.
    
    Attributes:
        env: The underlying Gymnasium environment
        render_mode: Rendering mode ('human' for visualization, None for no rendering)
        total_reward: Cumulative reward for the current episode
        total_steps: Number of steps taken in the current episode
        episode_history: List of dictionaries containing stats for completed episodes
    """
    
    def __init__(self, render_mode=None):
        """
        Initialize the CartPole wrapper.
        
        Args:
            render_mode: Optional rendering mode ('human' or None)
        """
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.render_mode = render_mode
        
        # Episode tracking
        self.total_reward = 0
        self.total_steps = 0
        self.episode_history = []
    
    def reset(self):
        """
        Reset the environment for a new episode.
        
        Returns:
            initial_observation: The starting state of the environment
        """
        # Save the previous episode's statistics
        if self.total_steps > 0:
            self.episode_history.append({
                "reward": self.total_reward,
                "steps": self.total_steps
            })
        
        # Reset counters
        self.total_reward = 0
        self.total_steps = 0
        
        # Reset the underlying environment
        initial_observation, info = self.env.reset()
        return initial_observation
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: The action to take (0 for left, 1 for right)
        
        Returns:
            observation: The new state after taking the action
            reward: The reward received
            terminated: Whether the episode ended due to a terminal condition
            truncated: Whether the episode ended due to time limit
            info: Additional information from the environment
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Update episode statistics
        self.total_reward += reward
        self.total_steps += 1
        
        # Render if requested
        if self.render_mode:
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment if render_mode is set."""
        if self.render_mode:
            self.env.render()
    
    def close(self):
        """Close the environment and clean up resources."""
        self.env.close()
    
    def get_state_space(self):
        """Return the observation space of the environment."""
        return self.env.observation_space
    
    def get_action_space(self):
        """Return the action space of the environment."""
        return self.env.action_space


class QLearningAgent:
    """
    A simple Q-learning agent for CartPole with state discretization.
    
    Since CartPole has a continuous state space, we discretize it into bins
    to use a Q-table approach.
    """
    
    def __init__(self, action_space, observation_space, bins=[10, 10, 10, 10], 
                 lr=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        """
        Initialize the Q-learning agent.
        
        Args:
            action_space: The action space of the environment
            observation_space: The observation space of the environment
            bins: Number of bins for discretizing each state dimension
            lr: Learning rate (alpha)
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon after each episode
            epsilon_min: Minimum epsilon value
        """
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Discretize the continuous observation space
        # CartPole state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        self.state_bins = [
            np.linspace(-2.4, 2.4, bins[0]),      # Cart position
            np.linspace(-4, 4, bins[1]),          # Cart velocity
            np.linspace(-0.2095, 0.2095, bins[2]), # Pole angle (±12 degrees)
            np.linspace(-4, 4, bins[3]),          # Pole angular velocity
        ]
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros(tuple(bins) + (action_space.n,))
    
    def discretize_state(self, state):
        """
        Convert a continuous state to a discrete state index.
        
        Args:
            state: Continuous state array
        
        Returns:
            Tuple of indices representing the discretized state
        """
        indices = []
        for i, val in enumerate(state):
            # Clip values to stay within bounds
            clipped_val = np.clip(val, self.state_bins[i][0], self.state_bins[i][-1])
            index = np.digitize(clipped_val, self.state_bins[i]) - 1
            # Ensure index is within valid range
            index = min(index, len(self.state_bins[i]) - 2)
            indices.append(index)
        return tuple(indices)
    
    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state
        
        Returns:
            Selected action (0 or 1)
        """
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            return self.action_space.sample()
        else:
            # Exploit: choose the best action from Q-table
            discrete_state = self.discretize_state(state)
            return np.argmax(self.q_table[discrete_state])
    
    def update_q_table(self, state, action, reward, next_state, done):
        """
        Update the Q-table using the Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        discrete_state = self.discretize_state(state)
        next_discrete_state = self.discretize_state(next_state)
        
        # Q-learning update rule
        old_value = self.q_table[discrete_state + (action,)]
        next_max = np.max(self.q_table[next_discrete_state])
        
        # Q(s,a) = Q(s,a) + lr * [reward + gamma * max(Q(s',a')) - Q(s,a)]
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[discrete_state + (action,)] = new_value
        
        # Decay epsilon after each episode
        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def run_random_policy(episodes=10, render=True):
    """
    Run episodes with a random policy to establish a baseline.
    
    Args:
        episodes: Number of episodes to run
        render: Whether to render the environment
    """
    print(f"\n{'='*60}")
    print("Running Random Policy Baseline")
    print(f"{'='*60}\n")
    
    render_mode = "human" if render else None
    env = CartPoleWrapper(render_mode=render_mode)
    
    for episode in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action = env.env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        print(f"Episode {episode + 1}: Reward = {env.episode_history[-1]['reward']:.0f}, "
              f"Steps = {env.episode_history[-1]['steps']}")
    
    # Save the final episode if not already saved
    if env.total_steps > 0:
        env.episode_history.append({
            "reward": env.total_reward,
            "steps": env.total_steps
        })
    
    env.close()
    
    # Calculate statistics
    rewards = [e["reward"] for e in env.episode_history]
    print(f"\nRandom Policy Statistics:")
    print(f"  Average Reward: {np.mean(rewards):.2f}")
    print(f"  Std Dev: {np.std(rewards):.2f}")
    print(f"  Min: {np.min(rewards):.0f}, Max: {np.max(rewards):.0f}")
    
    return env.episode_history


def train_agent(episodes=5000, render=False):
    """
    Train a Q-learning agent on CartPole.
    
    Args:
        episodes: Number of training episodes
        render: Whether to render the environment during training
    
    Returns:
        episode_history: List of episode statistics
    """
    print(f"\n{'='*60}")
    print("Training Q-Learning Agent")
    print(f"{'='*60}\n")
    
    render_mode = "human" if render else None
    env = CartPoleWrapper(render_mode=render_mode)
    agent = QLearningAgent(env.env.action_space, env.env.observation_space)
    
    for episode in range(episodes):
        obs = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Penalize failure to encourage longer episodes
            if done and env.total_steps < 500:
                reward = -10
            
            agent.update_q_table(obs, action, reward, next_obs, done)
            obs = next_obs
        
        # Print progress every 500 episodes
        if (episode + 1) % 500 == 0:
            recent_rewards = [e["reward"] for e in env.episode_history[-100:]]
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Avg Reward (last 100): {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    # Save the final episode if not already saved
    if env.total_steps > 0:
        env.episode_history.append({
            "reward": env.total_reward,
            "steps": env.total_steps
        })
    
    env.close()
    
    # Final statistics
    final_100 = [e["reward"] for e in env.episode_history[-100:]]
    print(f"\nTraining Complete!")
    print(f"  Final Average Reward (last 100 episodes): {np.mean(final_100):.2f}")
    print(f"  Final Epsilon: {agent.epsilon:.4f}")
    
    return env.episode_history


def plot_rewards(episode_history, window=100, filename="episode_rewards.png"):
    """
    Plot episode rewards over time with a moving average.
    
    Args:
        episode_history: List of episode statistics
        window: Window size for moving average
        filename: Output filename for the plot
    """
    rewards = [e["reward"] for e in episode_history]
    
    # Calculate moving average
    moving_avg = []
    for i in range(len(rewards)):
        start_idx = max(0, i - window + 1)
        moving_avg.append(np.mean(rewards[start_idx:i+1]))
    
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, label="Episode Reward")
    plt.plot(moving_avg, linewidth=2, label=f"Moving Average (window={window})")
    plt.title("CartPole Training Progress: Episode Rewards Over Time", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\nPlot saved to {filename}")
    plt.show()


def compare_policies():
    """
    Compare random policy vs trained policy performance.
    """
    print(f"\n{'='*60}")
    print("Comparing Random vs Trained Policy")
    print(f"{'='*60}\n")
    
    # Random policy
    print("1. Testing Random Policy...")
    random_history = run_random_policy(episodes=100, render=False)
    random_rewards = [e["reward"] for e in random_history]
    
    # Trained policy
    print("\n2. Training Agent...")
    trained_history = train_agent(episodes=3000, render=False)
    trained_rewards = [e["reward"] for e in trained_history[-100:]]  # Last 100 episodes
    
    # Comparison plot
    plt.figure(figsize=(10, 6))
    plt.boxplot([random_rewards, trained_rewards], labels=["Random Policy", "Trained Policy"])
    plt.title("Performance Comparison: Random vs Trained Policy", fontsize=14, fontweight='bold')
    plt.ylabel("Episode Reward", fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("policy_comparison.png", dpi=300)
    print(f"\nComparison plot saved to policy_comparison.png")
    plt.show()
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Performance Summary")
    print(f"{'='*60}")
    print(f"Random Policy:  Mean = {np.mean(random_rewards):.2f}, Std = {np.std(random_rewards):.2f}")
    print(f"Trained Policy: Mean = {np.mean(trained_rewards):.2f}, Std = {np.std(trained_rewards):.2f}")
    print(f"Improvement: {((np.mean(trained_rewards) - np.mean(random_rewards)) / np.mean(random_rewards) * 100):.1f}%")


def main():
    """
    Main function to run the complete tutorial workflow.
    """
    print("\n" + "="*60)
    print("CartPole Environment Wrapper Tutorial")
    print("="*60)
    
    # Option 1: Run random policy baseline
    # random_history = run_random_policy(episodes=10, render=True)
    
    # Option 2: Train agent and visualize
    history = train_agent(episodes=5000, render=False)
    plot_rewards(history)
    
    # Option 3: Compare policies
    # compare_policies()


if __name__ == "__main__":
    main()
