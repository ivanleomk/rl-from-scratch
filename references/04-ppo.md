
## Overview

Policy gradient methods (like REINFORCE and A2C) can be unstable. A single bad batch of data can lead to a large, destructive policy update, from which the agent may never recover. Proximal Policy Optimization (PPO) is a modern policy gradient algorithm that solves this problem by constraining how much the policy can change at each update.

## The Problem: Destructive Policy Updates

Imagine your agent has learned a good policy. Then, by chance, it gets a batch of data that suggests a very different, but actually worse, policy is good. If you take a large step in that direction, you can "fall off a cliff" and forget everything you learned.

This is the core problem with vanilla policy gradients: they are very sensitive to the step size (learning rate).

## The Solution: Trust Region Methods

**Trust Region Policy Optimization (TRPO)** was the first algorithm to formalize this idea. The core concept:

> Don't take a step that is too far away from your current policy. Only trust the gradient in a small region around your current policy.

TRPO enforces this with a complex second-order optimization constraint (based on KL divergence). It works well but is very complicated to implement.

## PPO: A Simpler Approach

**Proximal Policy Optimization (PPO)** achieves the same goal as TRPO but with a much simpler implementation. It's the default RL algorithm at many top labs for a reason: it's simple, stable, and effective.

### The PPO Clipped Objective

PPO introduces a new objective function that "clips" the policy update to keep it from changing too much.

Let `r(θ) = π_θ(a|s) / π_θ_old(a|s)` be the probability ratio between the new and old policies.

- If `r(θ) > 1`, the new policy is more likely to take that action.
- If `r(θ) < 1`, the new policy is less likely to take that action.

The standard policy gradient objective is `r(θ) * A`, where `A` is the advantage.

PPO modifies this:

`L_clip(θ) = E[min(r(θ) * A, clip(r(θ), 1 - ε, 1 + ε) * A)]`

Let's break this down:

- `ε` (epsilon) is a small hyperparameter, usually 0.2.
- `clip(r(θ), 1 - ε, 1 + ε)` clamps the ratio to be within `[0.8, 1.2]`.

**Case 1: Advantage `A` is positive** (the action was good)
- We want to increase `r(θ)`.
- The objective becomes `min(r(θ) * A, (1 + ε) * A)`.
- This means we can't increase the probability of this action by more than a factor of `1 + ε`.

**Case 2: Advantage `A` is negative** (the action was bad)
- We want to decrease `r(θ)`.
- The objective becomes `max(r(θ) * A, (1 - ε) * A)` (since `A` is negative).
- This means we can't decrease the probability of this action by more than a factor of `1 - ε`.

In short, the clipping prevents the policy from changing too drastically in a single update.

### The PPO Algorithm

1. Collect a batch of experience using the current policy `π_θ_old`.
2. For each step, compute the advantages `A_t` (using a value function, just like in A2C).
3. For `N` epochs, update the policy `π_θ` by optimizing the PPO clipped objective.
4. Update the value function by minimizing the MSE of the value predictions.
5. Repeat.

**Key difference from A2C**: PPO performs multiple epochs of updates on the same batch of data.

## Implementation Details

### Training Loop

```python
# Collect experience
states, actions, log_probs_old, returns, advantages = collect_experience()

# For N epochs
for _ in range(ppo_epochs):
    # Get new log probs from current policy
    log_probs_new = policy.get_log_probs(states, actions)
    
    # Compute ratio
    ratio = torch.exp(log_probs_new - log_probs_old)
    
    # Compute clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    
    # Compute value loss
    critic_loss = ...
    
    # Update
    loss = actor_loss + c * critic_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Hyperparameters

PPO is generally less sensitive to hyperparameters than other algorithms, but these are the key ones:

- `epsilon` (clip range): 0.1 or 0.2
- `ppo_epochs`: 4 to 10
- `num_minibatches`: Divide the batch into smaller minibatches for updates
- `learning_rate`: 3e-4 is a good starting point

## Common Issues & Debugging

### Issue: KL Divergence is too high

KL divergence measures how much the policy has changed. If it's too high, you're taking steps that are too large.

```python
kl_div = (log_probs_old - log_probs_new).mean().item()
```

**Fixes**:
- Decrease `epsilon`
- Decrease learning rate
- Add a KL penalty to the loss (an alternative PPO variant)

### Issue: Value function is not learning

**Symptom**: Critic loss is flat or increasing.

**Fixes**:
- Increase the critic loss coefficient `c`
- Use a separate optimizer for the critic

## Exercises

1. **Implement PPO from scratch**
   - Start with your A2C code
   - Modify the actor loss to use the PPO clipped objective
   - Add a loop for multiple PPO epochs
   - Test on a continuous control environment like Pendulum-v1

2. **Compare PPO, A2C, and REINFORCE**
   - Train all three on the same environment
   - Which is most stable? Which learns fastest?
   - PPO should be the most reliable.

3. **Experiment with `epsilon`**
   - Try `epsilon = 0.05`, `0.2`, `0.5`
   - How does it affect stability and learning speed?

## Key Takeaways

- PPO is a simple, stable, and high-performing RL algorithm.
- It prevents destructive policy updates by clipping the objective function.
- It's an on-policy algorithm but reuses data for multiple epochs, making it more sample-efficient than A2C.
- PPO is the go-to algorithm for many RL problems, especially in continuous control.

## Links & Resources

- [Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [TRPO Paper](https://arxiv.org/abs/1502.05477)
- [Blog: Understanding PPO](https://jonathan-hui.medium.com/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12)

## Next Steps

So far, we've focused on on-policy methods. In Topic 5, we'll switch gears to off-policy learning with DQN, which introduces experience replay and target networks.
