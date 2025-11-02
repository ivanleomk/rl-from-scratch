# Topic 3: Value Functions & Actor-Critic Methods

## Overview

REINFORCE (Topic 2) works, but it has high variance because the return `R_t` is a noisy estimate of an action's goodness. Actor-Critic methods reduce this variance by introducing a **value function baseline**.

## The Problem with REINFORCE

Imagine an environment where all rewards are positive. In REINFORCE, every action in a high-reward episode gets a positive update, even if some actions were bad. Conversely, every action in a low-reward episode gets a negative update, even if some were good.

This is inefficient. We want to know if an action was **better than average**, not just if it was part of a good episode.

## The Solution: Baselines

The policy gradient theorem can be modified to include a **baseline** `b(s)`:

```
∇J(θ) = E[∇log π(a|s) * (Q(s,a) - b(s))]
```

As long as the baseline `b(s)` does not depend on the action `a`, this is an unbiased estimate of the policy gradient. A good choice for the baseline is the **value function V(s)**.

### Value Function V(s)

The value function `V(s)` is the expected return from being in state `s` and following the current policy.

```
V(s) = E[R_t | s_t = s]
```

In plain English: "What's the average score I can expect from this state?"

### Advantage Function A(s,a)

By using `V(s)` as our baseline, we get the **advantage function**:

```
A(s,a) = Q(s,a) - V(s)
```

- `Q(s,a)`: Expected return after taking action `a` in state `s`
- `V(s)`: Average expected return from state `s`

So, `A(s,a)` tells us: "How much better is it to take action `a` than to act normally?"

- If `A(s,a) > 0`, action `a` was better than average.
- If `A(s,a) < 0`, action `a` was worse than average.

This is a much better learning signal than the raw return!

## Actor-Critic Architecture

This leads to the **Actor-Critic** architecture:

1. **The Actor**: The policy `π(a|s)`. It decides which action to take.
2. **The Critic**: The value function `V(s)`. It critiques the actor's actions.

Both are typically neural networks.

### The Training Loop

For each step:
1. **Actor**: Takes action `a_t` based on state `s_t`
2. **Environment**: Returns reward `r_t` and next state `s_{t+1}`
3. **Critic**: Computes the **TD Error** (Temporal Difference Error)
   ```
   δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
   ```
   This is our estimate of the advantage `A(s_t, a_t)`.

4. **Update the Critic**:
   - The critic's goal is to minimize the TD error.
   - Loss: `L_critic = δ_t^2` (mean squared error)
   - This trains the critic to be a better estimator of `V(s)`.

5. **Update the Actor**:
   - The actor's goal is to maximize expected reward.
   - Loss: `L_actor = -log π(a_t|s_t) * δ_t` (the policy gradient update)
   - We use the TD error `δ_t` as our advantage estimate.

### A2C: Advantage Actor-Critic

A2C is a synchronous, deterministic version of A3C (Asynchronous Advantage Actor-Critic). It's a standard, reliable algorithm.

**Key features**:
- The actor and critic often share lower layers of the neural network.
- We collect a batch of experience (e.g., 16 steps) before updating.
- We compute losses for both actor and critic and add them together:
  `L_total = L_actor + c * L_critic` (where `c` is a hyperparameter, often 0.5)

## Implementation Details

### Shared Network

```python
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(...)
        self.actor_head = nn.Linear(hidden_dim, num_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = self.base(state)
        action_probs = F.softmax(self.actor_head(x), dim=-1)
        value = self.critic_head(x)
        return action_probs, value
```

### Training Loop

```python
# Collect a batch of experience
for t in range(batch_size):
    # ... play one step ...
    log_probs.append(log_prob)
    values.append(value)
    rewards.append(reward)

# Compute returns and advantages
returns = compute_returns(rewards)
advantages = returns - values

# Compute losses
actor_loss = -(log_probs * advantages.detach()).mean()
critic_loss = advantages.pow(2).mean()

# Update
loss = actor_loss + 0.5 * critic_loss
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**Important**: We `detach()` the advantages when computing the actor loss. Why? Because we don't want to backpropagate through the value function when updating the actor. The actor should treat the advantages as fixed constants.

## Common Issues & Debugging

### Issue: Value function diverges

**Symptom**: Value loss goes to infinity.

**Fixes**:
- Lower the learning rate
- Check reward scale (normalize if necessary)
- Use gradient clipping

### Issue: Actor doesn't learn

**Symptom**: Actor loss stays flat, policy entropy doesn't decrease.

**Fixes**:
- Check if advantages are computed correctly
- Make sure you're using `advantages.detach()` in the actor loss
- Experiment with the critic loss coefficient `c`

## Exercises

1. **Implement A2C from scratch**
   - Start with a shared Actor-Critic network
   - Implement the training loop with batch updates
   - Test on CartPole-v1

2. **Compare with REINFORCE**
   - Train A2C and REINFORCE on the same environment
   - Plot their learning curves (reward vs. episodes)
   - Which learns faster? Which is more stable?
   - You should see that A2C has lower variance.

3. **Experiment with the critic loss coefficient**
   - Try `c = 0.1`, `c = 0.5`, `c = 1.0`
   - How does it affect learning?

## Key Takeaways

- Actor-Critic methods reduce variance by using a value function baseline.
- The **advantage** `A(s,a) = Q(s,a) - V(s)` is a better learning signal than raw returns.
- The **TD error** is a good estimate of the advantage.
- A2C is a simple, powerful, and widely used RL algorithm.
- Understanding Actor-Critic is essential for understanding more advanced methods like PPO and SAC.

## Links & Resources

- [Spinning Up: Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)
- [A3C Paper](https://arxiv.org/abs/1602.01783)
- [Lilian Weng: Actor-Critic Methods](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#actor-critic)

## Next Steps

A2C is great, but it can still be sensitive to step size. In Topic 4, we'll look at PPO, which adds a trust region to make training even more stable.
