"""
Tutorial 2: 1D GridWorld with Sparse Rewards

This script implements a custom 1D GridWorld environment using Gymnasium,
trains an agent using a simple Monte Carlo policy gradient method,
and visualizes the learned policy.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class GridWorld1D(gym.Env):
    """
    A simple 1D GridWorld environment.
    
    The agent starts at position 0 and must reach the goal at position size-1.
    Actions: 0 (left), 1 (right)
    Reward: +1 at goal, 0 everywhere else (sparse reward)
    """
    
    def __init__(self, size=10):
        """
        Initialize the GridWorld environment.
        
        Args:
            size (int): The size of the 1D grid
        """
        self.size = size
        self.agent_pos = 0
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(2)  # 0: left, 1: right
        self.observation_space = gym.spaces.Discrete(self.size)
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        
        Args:
            seed (int, optional): Random seed for reproducibility
            options (dict, optional): Additional options
            
        Returns:
            tuple: (initial_observation, info_dict)
        """
        super().reset(seed=seed)
        self.agent_pos = 0
        return self.agent_pos, {}
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (int): The action to take (0: left, 1: right)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Update agent position based on action
        if action == 0:  # Move left
            self.agent_pos = max(0, self.agent_pos - 1)
        elif action == 1:  # Move right
            self.agent_pos = min(self.size - 1, self.agent_pos + 1)
        
        # Check if goal is reached
        terminated = self.agent_pos == self.size - 1
        
        # Sparse reward: +1 only at the goal
        reward = 1 if terminated else 0
        
        truncated = False  # Not used in this simple environment
        
        return self.agent_pos, reward, terminated, truncated, {}
    
    def render(self):
        """
        Render the environment as a simple text-based visualization.
        """
        grid = ['-'] * self.size
        grid[self.agent_pos] = 'A'
        if self.agent_pos != self.size - 1:
            grid[self.size - 1] = 'G'
        print(" ".join(grid))


def test_environment():
    """
    Test the GridWorld environment with random actions.
    """
    print("=" * 60)
    print("Testing the GridWorld1D Environment")
    print("=" * 60)
    
    env = GridWorld1D()
    obs, info = env.reset()
    
    print("\nInitial state:")
    env.render()
    
    print("\nTaking 5 random actions:")
    for step in range(5):
        action = env.action_space.sample()
        action_name = "left" if action == 0 else "right"
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {step + 1}: Action = {action_name}")
        env.render()
        print(f"Reward: {reward}, Terminated: {terminated}")
        
        if terminated:
            print("Goal reached!")
            break
    
    print("\n" + "=" * 60)


def train_agent(env, n_episodes=1000, learning_rate=0.1, seed=42):
    """
    Train an agent using a simple Monte Carlo policy gradient method.
    
    Args:
        env: The GridWorld environment
        n_episodes (int): Number of training episodes
        learning_rate (float): Learning rate for policy updates
        seed (int): Random seed for reproducibility
        
    Returns:
        np.ndarray: The learned policy
    """
    np.random.seed(seed)
    
    # Initialize a uniform random policy
    policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    
    successful_episodes = 0
    
    print("\n" + "=" * 60)
    print("Training the Agent")
    print("=" * 60)
    print(f"Number of episodes: {n_episodes}")
    print(f"Learning rate: {learning_rate}")
    print(f"Random seed: {seed}\n")
    
    for episode in range(n_episodes):
        episode_history = []
        obs, info = env.reset()
        terminated = False
        
        # Run one episode
        while not terminated:
            # Select action based on current policy
            action_probs = policy[obs]
            action = np.random.choice(env.action_space.n, p=action_probs)
            
            # Store state-action pair
            episode_history.append((obs, action))
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
        
        # If the goal was reached, update the policy
        if reward == 1:
            successful_episodes += 1
            for state, action in episode_history:
                # Increase probability of the chosen action
                policy[state, action] += learning_rate * (1 - policy[state, action])
                # Normalize to ensure probabilities sum to 1
                policy[state] /= np.sum(policy[state])
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            success_rate = successful_episodes / (episode + 1) * 100
            print(f"Episode {episode + 1}/{n_episodes} - Success rate: {success_rate:.2f}%")
    
    final_success_rate = successful_episodes / n_episodes * 100
    print(f"\nTraining finished!")
    print(f"Total successful episodes: {successful_episodes}/{n_episodes} ({final_success_rate:.2f}%)")
    print("=" * 60)
    
    return policy


def visualize_policy(policy, env):
    """
    Visualize the learned policy as a bar chart.
    
    Args:
        policy (np.ndarray): The learned policy
        env: The GridWorld environment
    """
    # Extract the probability of moving right for each state
    prob_move_right = policy[:, 1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(env.observation_space.n), prob_move_right, color='skyblue', edgecolor='black')
    plt.xlabel('State (Position)', fontsize=12)
    plt.ylabel('Probability of Moving Right', fontsize=12)
    plt.title('Learned Policy: Probability of Moving Right in Each State', fontsize=14, fontweight='bold')
    plt.xticks(range(env.observation_space.n))
    plt.ylim([0, 1.1])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a horizontal line at 0.5 for reference
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random policy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('policy_visualization.png', dpi=300)
    print("\nPolicy visualization saved as 'policy_visualization.png'")
    plt.show()


def print_policy_table(policy, env):
    """
    Print the learned policy as a formatted table.
    
    Args:
        policy (np.ndarray): The learned policy
        env: The GridWorld environment
    """
    print("\n" + "=" * 60)
    print("Learned Policy Table")
    print("=" * 60)
    print(f"{'State':<10} {'P(Left)':<15} {'P(Right)':<15} {'Preferred Action':<20}")
    print("-" * 60)
    
    for state in range(env.observation_space.n):
        p_left = policy[state, 0]
        p_right = policy[state, 1]
        preferred = "Left" if p_left > p_right else "Right" if p_right > p_left else "Neutral"
        print(f"{state:<10} {p_left:<15.4f} {p_right:<15.4f} {preferred:<20}")
    
    print("=" * 60)


def demonstrate_learned_policy(env, policy):
    """
    Demonstrate the learned policy by running one episode.
    
    Args:
        env: The GridWorld environment
        policy (np.ndarray): The learned policy
    """
    print("\n" + "=" * 60)
    print("Demonstrating Learned Policy")
    print("=" * 60)
    
    obs, info = env.reset()
    terminated = False
    step_count = 0
    
    print("\nInitial state:")
    env.render()
    
    while not terminated and step_count < 20:  # Limit to 20 steps to avoid infinite loops
        action_probs = policy[obs]
        action = np.argmax(action_probs)  # Choose the action with highest probability
        action_name = "left" if action == 0 else "right"
        
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        print(f"\nStep {step_count}: Action = {action_name}")
        env.render()
        
        if terminated:
            print(f"\nGoal reached in {step_count} steps!")
            print(f"Total reward: {reward}")
    
    if not terminated:
        print("\nFailed to reach goal in 20 steps.")
    
    print("=" * 60)


def main():
    """
    Main function to run the complete tutorial.
    """
    # Create the environment
    env = GridWorld1D(size=10)
    
    # Test the environment
    test_environment()
    
    # Train the agent
    policy = train_agent(env, n_episodes=1000, learning_rate=0.1, seed=42)
    
    # Print the learned policy
    print_policy_table(policy, env)
    
    # Demonstrate the learned policy
    demonstrate_learned_policy(env, policy)
    
    # Visualize the policy
    visualize_policy(policy, env)


if __name__ == '__main__':
    main()
