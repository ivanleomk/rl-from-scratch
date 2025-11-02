# Spinning Up in RL: A Build-First Curriculum

A hands-on curriculum for learning Reinforcement Learning from scratch, bridging classical RL algorithms to modern LLM training with verifiers. This curriculum emphasizes **building and implementing** over theory, starting with toy environments and progressively scaling to real-world applications.

## Philosophy

This curriculum follows the principles from [OpenAI's Spinning Up](https://spinningup.openai.com/en/latest/spinningup/spinningup.html):

- **Write your own implementations** - Build algorithms from scratch to truly understand them
- **Simplicity is critical** - Start simple, add complexity gradually
- **Focus on understanding** - RL code fails silently; you need to know exactly what should happen
- **Measure everything** - Instrument your code extensively for debugging
- **Iterate fast** - Use simple environments first (CartPole, GridWorld) before scaling up

## Prerequisites

**Mathematics**: Probability (random variables, expected values), basic calculus (gradients)

**Programming**: Comfortable with Python, basic familiarity with PyTorch or TensorFlow

**Machine Learning**: Understanding of neural networks, backpropagation, and gradient descent

**Recommended Resources**:
- [Spinning Up: The Right Background](https://spinningup.openai.com/en/latest/spinningup/spinningup.html#the-right-background)
- [Spinning Up: Part 1 - Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

---

## Topic 1: Foundations - Toy Environments & Core Concepts

**Goal**: Build your first RL environments from scratch and understand the fundamental building blocks.

### What You'll Build
- A Number Guesser game (discrete actions, binary rewards)
- A 1D GridWorld (navigation, sparse rewards)
- A simple CartPole environment wrapper

### Core Concepts to Master
- **States, Actions, Rewards**: The basic RL loop
- **Binary vs. Continuous Rewards**: When to use each
- **Sparse vs. Dense Rewards**: The credit assignment problem
- **Episode, Trajectory, Return**: Terminology that matters

### Key Confusions to Resolve
- **Binary Rewards**: Why start with `{0, 1}` rewards? Because they're unambiguous. No partial credit means clearer debugging.
- **Reward Shaping**: How do you design rewards that actually teach the behavior you want?

### References
- [Spinning Up: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [OpenAI Gym Environments](https://gymnasium.farama.org/)
- See: `references/01-foundations.md`

---

## Topic 2: Your First Algorithm - REINFORCE (Vanilla Policy Gradient)

**Goal**: Implement the simplest policy gradient algorithm and understand advantages.

### What You'll Build
- A simple policy network (state → action probabilities)
- The REINFORCE algorithm (~80 lines)
- Training loop with logging and visualization

### Core Concepts to Master
- **Policy**: A function (neural network) that maps states to action probabilities
- **Policy Gradient**: The idea of directly optimizing the policy
- **Return (Reward-to-Go)**: Sum of future rewards from a given step
- **Advantage**: "Was this action better than average?"
- **Normalized Advantages**: Why we subtract mean and divide by std

### Key Confusions to Resolve
- **Why Normalize Advantages?**: Training stability. Without normalization, the scale of rewards can vary wildly between episodes, making gradients unstable. Normalization ensures you're always comparing "how much better than average" rather than absolute reward values.
- **Log Probabilities**: Why do we use `log_prob` in the loss? Because we're doing gradient ascent on expected reward.

### References
- [Spinning Up: Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html)
- [Spinning Up: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- Original Paper: [Policy Gradient Methods](http://www.cs.toronto.edu/~zemel/documents/411/rltutorial.pdf)
- See: `references/02-reinforce.md`

---

## Topic 3: Value Functions & Actor-Critic Methods

**Goal**: Understand value functions and implement A2C (Advantage Actor-Critic).

### What You'll Build
- A value function network (state → expected return)
- Actor-Critic architecture
- A2C algorithm implementation

### Core Concepts to Master
- **Value Function V(s)**: Expected return from a state
- **Q-Function Q(s,a)**: Expected return from a state-action pair
- **TD Error**: Difference between predicted and actual value
- **Baseline**: Using V(s) to reduce variance in policy gradients
- **Advantage = Q(s,a) - V(s)**: The true advantage formulation

### Key Confusions to Resolve
- **Why Do We Need a Value Function?**: Variance reduction. Pure policy gradients (REINFORCE) have high variance. A value function baseline reduces this.
- **Bias-Variance Tradeoff**: Introducing a value function adds bias but reduces variance.

### References
- [Spinning Up: Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)
- Paper: [Asynchronous Methods for Deep RL (A3C)](https://arxiv.org/abs/1602.01783)
- See: `references/03-actor-critic.md`

---

## Topic 4: Trust Regions & PPO

**Goal**: Understand why naive policy gradients can be unstable and implement PPO.

### What You'll Build
- PPO with clipped objective
- KL divergence monitoring
- Comparison experiments (REINFORCE vs. A2C vs. PPO)

### Core Concepts to Master
- **Policy Update Step Size**: Why large updates can be catastrophic
- **Trust Region**: Constraining how much the policy can change
- **KL Divergence**: Measuring policy change
- **Clipped Objective**: PPO's practical solution to trust regions

### Key Confusions to Resolve
- **Why Clip?**: If the policy changes too much in one update, it can "forget" what it learned and performance collapses. Clipping prevents this.
- **Old vs. New Policy**: PPO uses an "old" policy to collect data and a "new" policy being updated. The clip keeps them close.

### References
- [Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Spinning Up: TRPO](https://spinningup.openai.com/en/latest/algorithms/trpo.html)
- Paper: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- See: `references/04-ppo.md`

---

## Topic 5: Q-Learning & Off-Policy Methods

**Goal**: Understand value-based methods and implement DQN.

### What You'll Build
- Q-Network (state-action → value)
- Experience Replay buffer
- DQN algorithm with target networks

### Core Concepts to Master
- **Q-Learning**: Learning Q(s,a) directly
- **Off-Policy**: Learning from old experience
- **Experience Replay**: Breaking correlation in training data
- **Target Network**: Stabilizing Q-learning updates
- **Epsilon-Greedy Exploration**: Balancing exploration vs. exploitation

### Key Confusions to Resolve
- **Why Replay?**: On-policy methods (like PPO) must throw away data after one update. Off-policy methods can reuse it, making them more sample-efficient.
- **Why Target Networks?**: Without them, Q-learning is chasing a moving target (the Q-values keep changing as you update). Target networks stabilize this.

### References
- [Spinning Up: DQN](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#deep-q-learning)
- Paper: [Playing Atari with Deep RL](https://arxiv.org/abs/1312.5602)
- See: `references/05-dqn.md`

---

## Topic 6: Continuous Control & DDPG/TD3/SAC

**Goal**: Extend to continuous action spaces and implement modern actor-critic methods.

### What You'll Build
- DDPG (Deep Deterministic Policy Gradient)
- TD3 (Twin Delayed DDPG) improvements
- SAC (Soft Actor-Critic) with entropy regularization

### Core Concepts to Master
- **Continuous Actions**: When actions are real numbers, not discrete choices
- **Deterministic Policies**: Outputting a specific action instead of a probability distribution
- **Twin Critics**: TD3's solution to overestimation bias
- **Entropy Regularization**: Encouraging exploration in continuous spaces

### References
- [Spinning Up: DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
- [Spinning Up: TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html)
- [Spinning Up: SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)
- See: `references/06-continuous-control.md`

---

## Topic 7: Modern Context - LLMs & Verifiers

**Goal**: Connect classical RL to modern LLM training with process and outcome supervision.

### What You'll Build
- A simple verifier environment using the `verifiers` library
- Custom reward functions (rubrics)
- Comparison of outcome-based vs. process-based rewards

### Core Concepts to Master
- **LLM as Policy**: The language model is the agent
- **Text Generation as Actions**: Each token is an action
- **Verifiers as Reward Functions**: Automated grading of LLM outputs
- **Process Supervision**: Rewarding intermediate reasoning steps
- **Outcome Supervision**: Rewarding only the final answer
- **RLHF & RLAIF**: Human vs. AI feedback

### Key Confusions to Resolve
- **Why Verifiers?**: For complex tasks (coding, math), we can't hand-design reward functions. Verifiers automate this.
- **Process vs. Outcome**: Process supervision (rewarding each step) can prevent reward hacking and improve interpretability, but requires more annotation.

### References
- [PrimeIntellect Verifiers Repository](https://github.com/PrimeIntellect-ai/verifiers)
- [Verifiers Documentation](https://docs.verifiers.ai/)
- Paper: [Let's Verify Step by Step (Process Supervision)](https://arxiv.org/abs/2305.20050)
- Paper: [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
- See: `references/07-verifiers.md`

---

## Topic 8: Advanced Topics & Research Directions

**Goal**: Explore cutting-edge areas and develop research intuition.

### Areas to Explore
- **Exploration**: Intrinsic motivation, curiosity-driven learning
- **Model-Based RL**: Learning world models, planning
- **Multi-Agent RL**: Cooperation and competition
- **Hierarchical RL**: Learning skills and sub-policies
- **Meta-RL**: Learning to learn across tasks
- **Offline RL**: Learning from fixed datasets
- **Scaling RL**: Distributed training, large-scale experiments

### References
- [Spinning Up: Key Papers in Deep RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html)
- [Spinning Up: Developing a Research Project](https://spinningup.openai.com/en/latest/spinningup/spinningup.html#developing-a-research-project)
- See: `references/08-advanced-topics.md`

---

## Debugging & Best Practices

### Common Failure Modes
- **Silent Failures**: RL code often runs without errors but the agent never learns
- **Hyperparameter Sensitivity**: Small changes can break everything
- **Reward Hacking**: Agent finds unintended ways to maximize reward
- **Exploration Collapse**: Agent gets stuck in local optima

### Debugging Checklist
- Verify your environment is correct (test with random actions)
- Check reward scales and distributions
- Monitor policy entropy (are you exploring?)
- Visualize agent behavior (watch it play)
- Compare to known baselines
- Instrument everything (log mean/std/min/max of all key metrics)

### References
- [Spinning Up: Doing Rigorous Research in RL](https://spinningup.openai.com/en/latest/spinningup/spinningup.html#doing-rigorous-research-in-rl)
- See: `references/debugging-guide.md`

---

## Contributing

This curriculum is a living document. As you work through it, feel free to:
- Add your own implementations
- Create new exercises
- Improve explanations
- Add visualizations and notebooks

---

## Additional Resources

- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Sutton & Barto: Reinforcement Learning Book](http://incompleteideas.net/book/the-book-2nd.html)
- [David Silver's RL Course](https://www.davidsilver.uk/teaching/)
- [Berkeley CS 285: Deep RL](http://rail.eecs.berkeley.edu/deeprlcourse/)
- [Lilian Weng's RL Blog Posts](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)

---

**Let's build!**
