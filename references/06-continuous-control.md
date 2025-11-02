
## Overview

Many real-world problems (robotics, self-driving cars) have **continuous action spaces**, where actions are real-valued vectors (e.g., motor torques, steering angle). Algorithms like DQN and vanilla policy gradients don't work well here.

- **DQN**: `argmax_a Q(s,a)` is intractable if `a` is continuous.
- **Policy Gradients**: Sampling from a continuous distribution can have high variance.

This topic covers modern algorithms designed specifically for continuous control.

## DDPG: Deep Deterministic Policy Gradient

DDPG is an off-policy actor-critic algorithm that combines ideas from DQN and DPG (Deterministic Policy Gradient).

### Key Ideas

1. **Deterministic Actor**: Instead of a stochastic policy `π(a|s)` that outputs probabilities, DDPG uses a **deterministic policy** `μ(s)` that outputs a specific action.

2. **Off-Policy Learning**: Like DQN, DDPG uses an experience replay buffer and target networks to learn off-policy.

3. **Actor-Critic Architecture**:
   - **Critic `Q(s,a)`**: Learns the Q-function using the Bellman equation, just like in DQN.
   - **Actor `μ(s)`**: Learns a policy that maximizes `Q(s,a)`.

### The DDPG Update

**Critic Update**: The critic is updated just like in DQN, by minimizing the MSE loss:

- Target: `y = r + γ * Q_target(s’, μ_target(s’))`
- Loss: `L_critic = (y - Q(s,a))^2`

Notice that the target actor `μ_target` is used to select the next action.

**Actor Update**: The actor is updated by following the gradient of the Q-function with respect to the policy parameters. This is the "Deterministic Policy Gradient" part.

- Loss: `L_actor = -Q(s, μ(s))`

We want to find actions `μ(s)` that lead to high Q-values, so we minimize the negative Q-value.

### Exploration

Since the policy is deterministic, we need to add noise to the actions to encourage exploration.

`action = μ(s) + Noise`

Common choices for noise are Gaussian noise or an Ornstein-Uhlenbeck process.

## TD3: Twin Delayed DDPG

DDPG is notoriously unstable and sensitive to hyperparameters. TD3 (Twin Delayed DDPG) introduces three key improvements to stabilize it.

### Improvement 1: Clipped Double Q-Learning

**Problem**: DDPG suffers from **overestimation bias**. The `max` operator in the Bellman equation can lead to systematically overestimating Q-values.

**Solution**: Learn **two** Q-functions (critics) instead of one. When computing the target, use the minimum of the two Q-values.

- `Q_target = min(Q1_target(s’, a’), Q2_target(s’, a’))`

This helps to control the overestimation bias.

### Improvement 2: Delayed Policy Updates

**Problem**: The actor and critic can get into a feedback loop where a bad critic leads to a bad actor, which leads to a worse critic.

**Solution**: Update the actor **less frequently** than the critic. For example, update the critic every step, but update the actor and target networks every 2 steps.

This gives the critic time to converge to a better estimate before the actor is updated.

### Improvement 3: Target Policy Smoothing

**Problem**: The target values can be noisy due to function approximation error.

**Solution**: Add a small amount of clipped noise to the target action.

- `a’ = μ_target(s’) + clip(GaussianNoise, -c, c)`

This smooths out the Q-function landscape, making it more robust to errors.

## SAC: Soft Actor-Critic

SAC is another modern actor-critic algorithm that has become a state-of-the-art benchmark. It introduces the idea of **entropy regularization**.

### Maximum Entropy RL

The standard RL objective is to maximize expected reward. The maximum entropy RL objective is:

`J(π) = E[Σ(r_t + α * H(π(·|s_t)))]`

- `H(π(·|s_t))` is the entropy of the policy at state `s_t`.
- `α` (alpha) is a temperature parameter that controls the importance of the entropy term.

**Why this is useful**:
1. **Encourages exploration**: The agent is rewarded for acting randomly (high entropy).
2. **Improves robustness**: The agent learns multiple ways to solve a task, making it more robust to perturbations.
3. **More stable training**.

### SAC Algorithm

SAC is similar to TD3 but with a few key differences:

- It uses a **stochastic policy** (like PPO) instead of a deterministic one.
- It includes the entropy term in the value function updates.
- It learns the temperature parameter `α` automatically instead of treating it as a fixed hyperparameter.

SAC is known for its excellent performance and stability across a wide range of continuous control tasks.

## Implementation Details

- **Action Space Scaling**: Continuous actions often need to be scaled to a specific range (e.g., `[-1, 1]`). Use `tanh` on the output of the actor network.
- **Soft Target Updates**: Instead of hard-copying weights to the target network every `N` steps, use a soft update:
  `θ_target = τ * θ + (1 - τ) * θ_target`
  where `τ` (tau) is a small number like 0.005. This makes training more stable.

## Exercises

1. **Implement TD3 from scratch**
   - Start with a DDPG implementation and add the three TD3 improvements.
   - Test on a continuous control benchmark like Pendulum-v1 or BipedalWalker-v3.

2. **Implement SAC from scratch**
   - This is more challenging but very rewarding.
   - Pay close attention to the value function and policy updates, which include the entropy term.

3. **Compare TD3 and SAC**
   - Train both on the same set of environments.
   - Which learns faster? Which gets a higher final reward?
   - SAC is often more sample-efficient and stable.

## Key Takeaways

- Continuous control requires specialized algorithms.
- DDPG is the foundation, but it's unstable.
- **TD3** is a set of simple but powerful improvements that make DDPG practical.
- **SAC** is a state-of-the-art algorithm that uses entropy regularization for better exploration and stability.
- For most continuous control problems, TD3 or SAC are excellent starting points.

## Links & Resources

### DDPG
- [Spinning Up: DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
- [DDPG Paper](https://arxiv.org/abs/1509.02971)

### TD3
- [Spinning Up: TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html)
- [TD3 Paper](https://arxiv.org/abs/1802.09477)

### SAC
- [Spinning Up: SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)
- [SAC Paper](https://arxiv.org/abs/1801.01290)

## Next Steps

You now have a powerful toolkit of RL algorithms for both discrete and continuous action spaces. In Topic 7, we'll see how these classical algorithms are applied in the modern context of training Large Language Models.
