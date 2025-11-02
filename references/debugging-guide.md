# RL Debugging Guide

## The Golden Rule of RL Debugging

**RL code almost always fails silently.** The code runs without errors, but the agent never learns. This makes debugging uniquely challenging.

## Why RL is Hard to Debug

1. **No immediate feedback**: Unlike supervised learning, you don't know if your code is broken until after thousands of episodes
2. **Stochasticity**: Random seeds can make the same bug appear or disappear
3. **Hyperparameter sensitivity**: A small change in learning rate can mean the difference between working and not working
4. **Delayed consequences**: A bug in reward computation might not show up for hundreds of episodes

## The Debugging Checklist

Work through this checklist systematically when your agent isn't learning.

### Level 1: Environment Sanity Checks

Before you even touch your RL algorithm, verify your environment is correct.

**Test 1: Random Agent**
```python
env = YourEnvironment()
for episode in range(100):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = env.action_space.sample()  # Random action
        state, reward, done, info = env.step(action)
        total_reward += reward
    
    print(f"Episode {episode}: {total_reward}")
```

**What to check**:
- Does the environment ever give positive reward?
- Are the rewards in a reasonable range?
- Does the episode terminate correctly?
- Are there any crashes or errors?

**Test 2: Optimal Agent**

If possible, code up a simple optimal or near-optimal policy by hand.

```python
def optimal_policy(state):
    # For GridWorld: always move toward goal
    if state < goal:
        return "right"
    else:
        return "left"
```

What reward does the optimal policy get? This is your upper bound.

**Test 3: Visualization**

Watch your agent play. This is non-negotiable.

```python
env.render()  # If available
# OR
print(f"State: {state}, Action: {action}, Reward: {reward}")
```

If you can't see what's happening, you can't debug it.

### Level 2: Algorithm Sanity Checks

**Test 4: Verify Gradient Flow**

```python
# Before training loop
for param in policy.parameters():
    print(param.requires_grad)  # Should all be True

# After loss.backward()
for name, param in policy.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean()}")
    else:
        print(f"{name}: NO GRADIENT!")
```

If gradients are zero or None, your loss isn't connected to your parameters.

**Test 5: Overfit to One Episode**

Can your agent memorize a single episode?

```python
# Generate one episode
states, actions, rewards = play_episode(env, policy)

# Train on it repeatedly
for _ in range(1000):
    loss = compute_loss(states, actions, rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

If it can't overfit to one episode, something is fundamentally broken.

**Test 6: Check Advantage Computation**

```python
advantages = compute_advantages(rewards)
print(f"Advantages: {advantages}")
print(f"Mean: {advantages.mean()}, Std: {advantages.std()}")

# After normalization
normalized = normalize(advantages)
print(f"Normalized: {normalized}")
print(f"Mean: {normalized.mean()}, Std: {normalized.std()}")
```

After normalization, mean should be ~0 and std should be ~1.

### Level 3: Training Diagnostics

**Metrics to Log (Minimum)**:

```python
# Every episode
- episode_reward (mean, std, min, max)
- episode_length
- policy_loss
- value_loss (if using critic)

# Every N episodes
- policy_entropy (are you exploring?)
- gradient_norm (are gradients exploding?)
- learning_rate (if using scheduler)
```

**Test 7: Learning Curve Analysis**

Plot episode reward over time. What do you see?

- **Flat line**: Not learning at all
  - Check reward scale
  - Check learning rate
  - Check if gradients are flowing

- **Noisy but trending up**: Normal! RL is noisy.
  - Use smoothing (rolling average)
  - Increase batch size to reduce noise

- **Increases then crashes**: Catastrophic forgetting
  - Reduce learning rate
  - Use PPO instead of vanilla policy gradient
  - Check for bugs in advantage normalization

- **Oscillating wildly**: Unstable training
  - Normalize advantages
  - Reduce learning rate
  - Increase batch size

**Test 8: Policy Entropy**

```python
probs = policy(state)
entropy = -(probs * torch.log(probs + 1e-8)).sum()
```

- **High entropy (close to log(num_actions))**: Policy is random (exploring)
- **Low entropy (close to 0)**: Policy is deterministic (exploiting)

You want entropy to start high and gradually decrease. If it drops to zero immediately, your agent stopped exploring too early.

### Level 4: Hyperparameter Debugging

**Common Culprits**:

1. **Learning Rate**
   - Too high: Training is unstable, loss oscillates
   - Too low: Training is too slow, appears to not learn
   - Try: 0.001, 0.0003, 0.0001

2. **Batch Size**
   - Too small: High variance, unstable training
   - Too large: Slow iteration, might overfit
   - Try: 10-50 episodes for policy gradients

3. **Discount Factor (gamma)**
   - Too low (< 0.9): Agent is too short-sighted
   - Too high (> 0.99): Rewards are too delayed
   - Try: 0.99 for most tasks

4. **Network Architecture**
   - Too small: Can't learn complex policies
   - Too large: Overfits, slow training
   - Try: 2-3 hidden layers, 64-128 units each

## Common Bugs & How to Find Them

### Bug: Reward Scale is Wrong

**Symptom**: Agent never learns, or learning is extremely slow.

**Diagnosis**:
```python
print(f"Reward range: {min(rewards)} to {max(rewards)}")
```

**Fix**: Normalize rewards to roughly [-1, 1] or [0, 1].

### Bug: Forgot to Normalize Advantages

**Symptom**: Training is unstable, loss oscillates wildly.

**Diagnosis**:
```python
print(f"Advantage range: {advantages.min()} to {advantages.max()}")
```

**Fix**: Always normalize advantages.

### Bug: Using Old Policy for Actions

**Symptom**: Policy doesn't improve, or improves very slowly.

**Diagnosis**: Make sure you're using the updated policy to collect new data, not an old cached version.

**Fix**:
```python
# Wrong
old_policy = policy
# ... training ...
actions = old_policy(states)  # BUG: using old policy

# Right
actions = policy(states)  # Always use current policy
```

### Bug: Incorrect Advantage Computation

**Symptom**: Agent learns the opposite of what you want.

**Diagnosis**:
```python
# Print advantages for a known good episode
print(f"Good episode advantages: {advantages}")
# Should be positive

# Print advantages for a known bad episode
print(f"Bad episode advantages: {advantages}")
# Should be negative (after normalization)
```

**Fix**: Double-check your return computation. Are you summing rewards correctly?

### Bug: Gradient Clipping Too Aggressive

**Symptom**: Training is stable but very slow.

**Diagnosis**:
```python
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
print(f"Gradient norm: {total_norm}")
```

If gradients are always being clipped, your max_norm might be too low.

**Fix**: Increase max_norm or remove clipping entirely.

## The Scientific Method for RL Debugging

1. **Hypothesis**: What do you think is broken?
2. **Prediction**: If that's broken, what would you observe?
3. **Test**: Run an experiment to check your prediction
4. **Analyze**: Did the results match your prediction?
5. **Iterate**: Form a new hypothesis based on results

## When to Ask for Help

You should debug on your own first, but ask for help if:

1. You've worked through the entire checklist
2. You've tested on a simpler environment and it works there
3. You've compared your code to a reference implementation line-by-line
4. You've been stuck for more than a day

When asking for help, provide:
- Your environment code
- Your algorithm code
- Learning curves (plots)
- What you've already tried

## Tools & Techniques

### Logging

Use a proper logging framework:

```python
import wandb  # or tensorboard

wandb.init(project="my-rl-project")

for episode in range(num_episodes):
    # ... training ...
    wandb.log({
        "episode_reward": total_reward,
        "policy_loss": loss.item(),
        "entropy": entropy.item(),
    })
```

### Visualization

```python
import matplotlib.pyplot as plt

plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Learning Curve")
plt.savefig("learning_curve.png")
```

### Reproducibility

```python
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)
```

## Final Advice

1. **Start simple**: Test on the simplest possible environment first
2. **Change one thing at a time**: Don't tweak multiple hyperparameters simultaneously
3. **Trust the math**: If the theory says it should work, the bug is in your implementation
4. **Visualize everything**: Plots and videos are your best debugging tools
5. **Be patient**: RL is hard. Even experts spend days debugging.

## Resources

- [Spinning Up: Doing Rigorous Research in RL](https://spinningup.openai.com/en/latest/spinningup/spinningup.html#doing-rigorous-research-in-rl)
- [Deep RL Debugging Guide (Andrej Karpathy)](http://karpathy.github.io/2016/09/07/phd/)
- [37 Reasons Why Your RL Agent Doesn't Work](https://andyljones.com/posts/rl-debugging.html)
