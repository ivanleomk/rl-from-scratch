
## Overview

Before implementing any RL algorithms, you need to deeply understand what an RL environment is and how the agent-environment interaction loop works. The best way to learn this is by building your own environments from scratch.

## Key Concepts

### The RL Loop

At every timestep:
1. Agent observes **state** `s_t`
2. Agent takes **action** `a_t`
3. Environment returns **reward** `r_t` and new **state** `s_{t+1}`
4. Episode ends when **done** = True

### State Space

The **state** is all the information the agent needs to make a decision. In a grid world, this might be the agent's position. In a game, it might be the pixel values of the screen.

**Key Question**: What information does the agent actually need? Too much information makes learning harder. Too little makes the task impossible.

### Action Space

**Discrete Actions**: A finite set of choices (e.g., {Left, Right, Up, Down})

**Continuous Actions**: Real-valued vectors (e.g., steering angle and throttle in a car)

Start with discrete actions - they're simpler to implement and debug.

### Reward Design

This is the hardest part of RL. The reward function defines what "success" means.

**Binary Rewards**: `{0, 1}` - Either you succeeded or you didn't. Simple and unambiguous.

**Sparse Rewards**: Reward only comes at the end of an episode (e.g., +1 for winning a game). Hard to learn from because of the credit assignment problem.

**Dense Rewards**: Reward at every step (e.g., +1 for moving closer to the goal). Easier to learn from, but can lead to reward hacking.

**Shaped Rewards**: Carefully designed rewards that guide the agent toward the goal. Powerful but requires domain knowledge.

### Common Pitfalls

**Reward Hacking**: The agent finds an unintended way to maximize reward. Classic example: A cleaning robot that makes a mess so it can get reward for cleaning it up.

**Sparse Reward Problem**: If the agent only gets reward at the very end, it's like searching for a needle in a haystack. Random exploration might never find the goal.

## What to Build

### 1. Number Guesser

A simple game where the agent tries to guess a secret number in a limited number of attempts.

**Why this environment?**
- Extremely simple to implement
- Teaches you the basic environment API
- Binary rewards are unambiguous
- Discrete action space

**Implementation checklist**:
- `reset()` method that initializes a new episode
- `step(action)` method that returns `(state, reward, done, info)`
- Proper state representation
- Clear terminal condition

### 2. 1D GridWorld

A one-dimensional grid where the agent starts at one end and must reach the other.

**Why this environment?**
- Introduces spatial reasoning
- Demonstrates sparse rewards
- Shows the credit assignment problem
- Still simple enough to debug easily

**Implementation checklist**:
- Grid representation
- Agent position tracking
- Movement actions (left, right)
- Boundary handling
- Goal detection

### 3. CartPole Wrapper

Wrap OpenAI Gym's CartPole environment to understand the standard interface.

**Why this environment?**
- Industry-standard benchmark
- Continuous state space (position, velocity, angle, angular velocity)
- Teaches you to work with existing environments
- Fast iteration for testing algorithms

## Exercises

1. **Implement Number Guesser**: Write the environment from scratch. Test it with random actions. Does it behave correctly?

2. **Reward Experiments**: Try different reward schemes in your GridWorld:
   - Binary: +1 at goal, 0 elsewhere
   - Sparse: +1 at goal, -0.01 per step (encourages speed)
   - Dense: +1 for moving toward goal, -1 for moving away
   
   Which is easiest to learn from? Which leads to the best final policy?

3. **Visualization**: Create a simple visualization of your agent playing. This is crucial for debugging.

## Key Takeaways

- The environment defines the task. Get this right before worrying about algorithms.
- Reward design is critical and often harder than the algorithm itself.
- Binary rewards are a good starting point because they're unambiguous.
- Always test your environment with random actions first.
- Visualization is not optional - you need to see what the agent is doing.

## Links & Resources

- [OpenAI Gym Documentation](https://gymnasium.farama.org/)
- [Spinning Up: Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [Sutton & Barto Chapter 3: Finite MDPs](http://incompleteideas.net/book/the-book-2nd.html)

## Next Steps

Once you have a working environment and can visualize random agent behavior, you're ready to implement your first learning algorithm: REINFORCE (Topic 2).
