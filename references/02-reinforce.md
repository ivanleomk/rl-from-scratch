
## Overview

REINFORCE is the simplest policy gradient algorithm. It directly optimizes the policy by following the gradient of expected reward. Despite its simplicity, understanding REINFORCE deeply will give you intuition for all modern policy gradient methods (PPO, TRPO, etc.).

## The Core Idea

Instead of learning which actions are good (like Q-learning), we directly learn a policy that outputs action probabilities. Then we adjust the policy to make good actions more likely.

**The Policy Gradient Theorem** tells us how to compute the gradient of expected reward with respect to policy parameters:

```
∇J(θ) = E[∇log π(a|s) * R]
```

In plain English: "Make actions that led to high rewards more likely."

## Key Concepts

### Policy Network

A neural network that takes a state as input and outputs a probability distribution over actions.

For discrete actions:
- Input: State vector
- Hidden layers: A few fully connected layers with ReLU
- Output: Softmax over action space

**Why softmax?** Because we need valid probabilities (sum to 1, all positive).

### Log Probability

We use `log π(a|s)` instead of `π(a|s)` for numerical stability and because it makes the math cleaner.

When we sample an action from the policy, we also compute its log probability. This is what we'll use to compute gradients.

### Return (Reward-to-Go)

The **return** at time step `t` is the sum of all future rewards:

```
R_t = r_t + r_{t+1} + r_{t+2} + ... + r_T
```

This is also called "reward-to-go" because it's the total reward you'll get from this point onward.

**Why not just use the immediate reward `r_t`?** Because actions have long-term consequences. An action might have no immediate reward but lead to a big reward later.

### Advantages and Why We Normalize

The **advantage** tells us: "Was this action better or worse than average?"

In REINFORCE, we use the return as a simple advantage estimate. But returns can vary wildly in scale:
- Episode 1: Total return = 10
- Episode 2: Total return = 100
- Episode 3: Total return = 5

If we use these raw values, the gradients will be dominated by episode 2. This makes training unstable.

**Solution: Normalize advantages**

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

Now we're asking: "How much better than average was this action *in this batch*?"

This is crucial for training stability. Without normalization, your policy will oscillate wildly and may never converge.

### The REINFORCE Update

For each episode:
1. Collect trajectory: `(s_0, a_0, r_0), (s_1, a_1, r_1), ..., (s_T, a_T, r_T)`
2. Compute returns: `R_t = sum of rewards from t to T`
3. Normalize returns across the batch
4. Compute loss: `loss = -sum(log_prob * normalized_return)`
5. Backpropagate and update policy

**Why the negative sign?** We want to maximize reward, but optimizers minimize loss. So we minimize negative reward.

## Implementation Details

### Sampling Actions

```python
# Get action probabilities from policy
probs = policy(state)

# Sample an action
action = torch.multinomial(probs, 1)

# Compute log probability (needed for gradient)
log_prob = torch.log(probs[action])
```

### Computing Returns

```python
def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns
```

**The discount factor `gamma`**: Usually set to 0.99. This means future rewards are worth slightly less than immediate rewards. Why?
1. Uncertainty: The future is uncertain
2. Encourages faster solutions
3. Mathematical convenience (ensures finite returns)

### The Training Loop

```python
for episode in range(num_episodes):
    # Collect batch of episodes
    batch_log_probs = []
    batch_returns = []
    
    for _ in range(batch_size):
        states, actions, rewards = play_episode(env, policy)
        returns = compute_returns(rewards)
        
        batch_log_probs.extend(log_probs)
        batch_returns.extend(returns)
    
    # Normalize returns
    returns = normalize(batch_returns)
    
    # Compute loss and update
    loss = -(log_probs * returns).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Common Issues & Debugging

### Issue: Agent Never Learns

**Possible causes**:
1. Reward scale is wrong (too small or too large)
2. Learning rate is wrong
3. Not collecting enough episodes per batch
4. Bug in return computation
5. Forgot to normalize advantages

**Debug steps**:
- Print average return per episode. Is it increasing?
- Print policy entropy. Is it decreasing over time? (Should start high, end low)
- Visualize the agent. Is it doing anything sensible?
- Test on a trivial environment first (like Number Guesser)

### Issue: Learning is Unstable

**Possible causes**:
1. Not normalizing advantages
2. Learning rate too high
3. Batch size too small

**Solutions**:
- Always normalize advantages
- Try learning rate = 0.01 or 0.001
- Use batch_size >= 10 episodes

### Issue: Agent Learns Then Forgets

This is called "catastrophic forgetting" and is common in policy gradients.

**Why it happens**: A big policy update can make the agent forget what it learned.

**Solution**: Use smaller learning rates, or move to PPO (Topic 4) which explicitly prevents this.

## Exercises

1. **Implement REINFORCE from scratch** (~80 lines of code)
   - Start with Number Guesser environment
   - Implement the policy network
   - Implement return computation
   - Implement the training loop
   - Add logging (average return per episode)

2. **Experiment with normalization**
   - Train with normalized advantages
   - Train without normalization
   - Compare learning curves
   - Which converges faster? Which is more stable?

3. **Experiment with discount factor**
   - Try gamma = 1.0 (no discounting)
   - Try gamma = 0.99
   - Try gamma = 0.9
   - How does it affect learning speed and final performance?

4. **Visualize the policy**
   - After training, visualize what the policy has learned
   - For each state, what action does it prefer?
   - Does it make sense?

## Key Takeaways

- REINFORCE is simple but powerful: directly optimize what you care about (expected reward)
- Advantages tell you if an action was better than average
- Normalizing advantages is critical for stable training
- High variance is REINFORCE's main weakness (solved by adding a baseline in Actor-Critic)
- Always visualize your agent's behavior - it's the best debugging tool

## Mathematical Intuition

The policy gradient theorem says:

```
∇J(θ) = E[∇log π(a|s) * Q(s,a)]
```

In REINFORCE, we estimate `Q(s,a)` with the return `R_t`. This is an unbiased estimate, but it has high variance because:
- Returns are noisy (randomness in environment)
- We're using a single sample to estimate an expectation

This high variance is why we need:
1. Large batch sizes
2. Normalized advantages
3. Eventually, a baseline (Actor-Critic, next topic)

## Links & Resources

- [Spinning Up: Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html)
- [Spinning Up: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- [Policy Gradient Theorem Proof](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#policy-gradient-theorem)
- [Sutton & Barto Chapter 13: Policy Gradient Methods](http://incompleteideas.net/book/the-book-2nd.html)

## Next Steps

REINFORCE works, but it has high variance. In Topic 3, we'll add a **value function baseline** to reduce variance and create Actor-Critic methods.
