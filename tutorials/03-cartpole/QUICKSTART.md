# Quick Start Guide

## Installation

This tutorial uses `uv` for package management, but you can also use `pip`.

### Option 1: Using uv (recommended)
```bash
# Install dependencies
uv pip install -e .
```

### Option 2: Using pip
```bash
# Install dependencies
pip install numpy matplotlib gymnasium
```

## Running the Tutorial

### 1. Test the Wrapper with Random Policy
```bash
python cartpole_wrapper.py
```
This will train a Q-learning agent for 5000 episodes and generate a plot showing the learning progress.

### 2. Run Different Modes

Edit the `main()` function in `cartpole_wrapper.py` to uncomment different options:

**Option 1: Random Policy Baseline**
```python
def main():
    random_history = run_random_policy(episodes=10, render=True)
```

**Option 2: Train Agent (default)**
```python
def main():
    history = train_agent(episodes=5000, render=False)
    plot_rewards(history)
```

**Option 3: Compare Policies**
```python
def main():
    compare_policies()
```

## Expected Output

After training, you should see:
- Console output showing training progress every 500 episodes
- A plot file `episode_rewards.png` showing the learning curve
- Final statistics including average reward over the last 100 episodes

A well-trained agent should achieve an average reward of 200+ (with some reaching the maximum of 500).

## Understanding the Results

- **Random Policy**: Typically achieves 15-25 reward per episode
- **Trained Policy**: Should achieve 150-300+ reward per episode after 5000 episodes
- The CartPole-v1 environment is considered "solved" when the agent achieves an average reward of 475 or higher over 100 consecutive episodes

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'gymnasium'`
- Solution: Install dependencies using one of the methods above

**Issue**: Training is too slow
- Solution: Reduce the number of episodes in `train_agent(episodes=1000)`

**Issue**: No visualization window appears
- Solution: This is normal when `render=False`. Set `render=True` to see the CartPole animation, but note this will slow down training significantly.

## Next Steps

1. Experiment with different hyperparameters in the `QLearningAgent` class
2. Try different discretization bin sizes
3. Implement a different learning algorithm (e.g., SARSA, DQN)
4. Modify the reward structure to see how it affects learning

Happy learning!
