
## Overview

Everything you've learned about classical RL (policies, rewards, advantages, PPO) directly applies to training Large Language Models. The key insight: **an LLM is just a policy that outputs text instead of game actions**.

This topic connects the dots between the algorithms you've implemented and the cutting-edge application of RL to LLM training.

## The Core Mapping

| Classical RL (GridWorld) | LLM Training |
|-------------------------|--------------|
| **State** | The text generated so far (prompt + partial response) |
| **Action** | The next token to generate |
| **Policy** | The LLM itself (maps text → probability distribution over next tokens) |
| **Reward** | Score from a verifier (0.0 to 1.0) judging the quality of the output |
| **Episode** | Generating a complete response to a prompt |
| **Trajectory** | The sequence of tokens generated |

## What is a Verifier?

A **verifier** is an automated reward function for LLM outputs. Instead of hand-coding rewards (like "+1 for reaching the goal"), we use:

1. **Rule-based verifiers**: Check if the output satisfies specific criteria (e.g., "Does it contain valid JSON?")
2. **Model-based verifiers**: Another model that scores the output (e.g., "How helpful is this response?")
3. **Execution-based verifiers**: Run code and check if it passes tests

### Why Verifiers?

For complex tasks (coding, math, reasoning), we can't hand-design reward functions. Verifiers automate this:

- **Math problems**: Does the final answer match the correct solution?
- **Code generation**: Does the code pass all test cases?
- **Reasoning**: Does the logic lead to the correct conclusion?

## Process vs. Outcome Supervision

This is a critical distinction in modern LLM training.

### Outcome-Based Rewards (Outcome Supervision)

Reward only the final result.

**Example (Math Problem)**:
- Prompt: "What is 2 + 2?"
- Response: "First, I'll add 1 + 1 to get 2. Then I'll add 2 + 2 to get 5."
- Reward: **0** (final answer is wrong)

**Pros**:
- Simple to implement
- Only need to label final answers

**Cons**:
- No credit for correct intermediate steps
- Agent might learn shortcuts or reward hacking
- Hard to debug when things go wrong

### Process-Based Rewards (Process Supervision)

Reward each intermediate reasoning step.

**Example (Math Problem)**:
- Prompt: "What is 2 + 2?"
- Step 1: "First, I'll add 1 + 1 to get 2." → Reward: **0.5** (correct calculation, but not relevant)
- Step 2: "Then I'll add 2 + 2 to get 4." → Reward: **1.0** (correct!)

**Pros**:
- Rewards correct reasoning, not just correct answers
- Reduces reward hacking
- Better credit assignment (know which step went wrong)
- More interpretable

**Cons**:
- Requires labeling every step (expensive)
- More complex to implement

### When to Use Each

**Use Outcome Supervision when**:
- You only care about the final result
- Labeling intermediate steps is too expensive
- The task is simple enough that shortcuts aren't a problem

**Use Process Supervision when**:
- You want to encourage specific reasoning patterns
- Reward hacking is a concern
- You need interpretability (e.g., safety-critical applications)
- You have the resources to label intermediate steps

## The Verifiers Library

[PrimeIntellect's Verifiers](https://github.com/PrimeIntellect-ai/verifiers) is a framework for creating RL environments for LLM training.

### Core Components

1. **Dataset**: A collection of prompts (and optionally, ground truth answers)
2. **Rollout Logic**: How the environment interacts with the LLM (single-turn or multi-turn)
3. **Rubric**: A set of reward functions that score the LLM's output
4. **Parser**: Optional logic for extracting structured information from text

### Example: A Simple Verifier Environment

```python
import verifiers as vf
from datasets import load_dataset

# Load a dataset of math problems
dataset = load_dataset("my-account/math-problems", split="train")

# Define a reward function
def correctness_reward(prompt, completion, info):
    """Check if the final answer matches the ground truth."""
    predicted_answer = extract_answer(completion)
    true_answer = info["answer"]
    return 1.0 if predicted_answer == true_answer else 0.0

# Create a rubric (set of reward functions)
rubric = vf.Rubric(
    funcs=[correctness_reward],
    weights=[1.0]
)

# Create the environment
env = vf.SingleTurnEnv(
    dataset=dataset,
    rubric=rubric
)

# Evaluate an LLM
results = env.evaluate_sync(
    client=OpenAI(),
    model="gpt-4.1-mini",
    num_examples=100,
    rollouts_per_example=1
)
```

### Multi-Turn Environments

For more complex tasks (like coding or interactive problem-solving), you need multi-turn environments where the LLM can interact with the environment over multiple steps.

```python
class CodingEnv(vf.MultiTurnEnv):
    def env_response(self, state, completion):
        """Run the generated code and return test results."""
        code = extract_code(completion)
        test_results = run_tests(code)
        return test_results
    
    def is_completed(self, state):
        """Episode ends when all tests pass or max turns reached."""
        return all_tests_passed(state) or state["turn"] >= 5
```

## Training with Verifiers

Once you have a verifier environment, you can train an LLM with RL algorithms (PPO, GRPO, etc.).

The Verifiers library includes:
- **RLTrainer**: A simple trainer for small-scale experiments
- **prime-rl**: Large-scale distributed training with FSDP

### Training Loop (Conceptual)

```python
# 1. Collect rollouts
for prompt in dataset:
    response = llm.generate(prompt)
    reward = verifier.score(prompt, response)
    store_trajectory(prompt, response, reward)

# 2. Compute advantages
advantages = compute_advantages(rewards)
advantages = normalize(advantages)

# 3. Update policy (LLM) with PPO
for epoch in range(ppo_epochs):
    loss = compute_ppo_loss(trajectories, advantages)
    optimizer.step()
```

This is exactly the same as training on GridWorld, just with text instead of grid positions!

## Practical Considerations

### Reward Hacking

LLMs are very good at finding shortcuts. Common examples:

- **Verbosity**: Agent learns to generate long responses because length correlates with reward
- **Repetition**: Agent repeats the prompt back to get easy points
- **Format gaming**: Agent learns to output text that looks right but is meaningless

**Solutions**:
- Use process supervision to reward correct reasoning
- Add auxiliary rewards (e.g., penalize length, reward diversity)
- Use multiple verifiers to cross-check

### Exploration vs. Exploitation

LLMs have a natural exploration mechanism: **sampling temperature**.

- High temperature → More random, more exploration
- Low temperature → More deterministic, more exploitation

During training, you typically:
1. Start with higher temperature (explore)
2. Gradually decrease temperature (exploit)

### Sample Efficiency

LLM training is expensive. Each rollout requires generating text, which is slow.

**Strategies for efficiency**:
- Use smaller models for early experiments
- Use off-policy algorithms (can reuse old data)
- Use rejection sampling (only train on good examples)

## Exercises

1. **Install and Explore Verifiers**
   ```bash
   pip install verifiers
   prime env install will/wordle
   vf-eval wordle -m gpt-4.1-mini
   ```
   - Examine the Wordle environment code
   - What is the reward function?
   - Is it outcome-based or process-based?

2. **Create a Custom Verifier**
   - Pick a simple task (e.g., "Generate a number between 1 and 10")
   - Write a reward function that scores the output
   - Test it with an API-based LLM

3. **Compare Outcome vs. Process Supervision**
   - Create two versions of a math problem environment
   - One with outcome-only rewards
   - One with step-by-step rewards
   - Which learns faster? Which produces better reasoning?

4. **Analyze Reward Hacking**
   - Create a verifier that rewards "helpful" responses
   - Train a small model with it
   - Does it find any shortcuts?
   - How would you fix them?

## Key Takeaways

- LLMs are policies. Text generation is action selection.
- Verifiers are automated reward functions for complex tasks.
- Process supervision rewards reasoning steps, not just final answers.
- Outcome supervision is simpler but more prone to reward hacking.
- The RL algorithms you learned (PPO, A2C) are directly used for LLM training.
- Reward design is still the hardest part, even with verifiers.

## Research Frontiers

Current open questions in verifier-based LLM training:

1. **Scalable Process Supervision**: How do we label intermediate steps efficiently?
2. **Verifier Robustness**: How do we prevent LLMs from fooling verifiers?
3. **Multi-Objective RL**: How do we balance multiple rewards (correctness, safety, helpfulness)?
4. **Offline RL for LLMs**: Can we learn from fixed datasets of human text?
5. **Exploration in Text Space**: How do we explore effectively in the huge space of possible text?

## Links & Resources

### Verifiers Library
- [PrimeIntellect Verifiers Repository](https://github.com/PrimeIntellect-ai/verifiers)
- [Verifiers Documentation](https://docs.verifiers.ai/)
- [Environments Hub](https://hub.primeintellect.ai/)

### Key Papers
- [Training Verifiers to Solve Math Word Problems (OpenAI, 2021)](https://arxiv.org/abs/2110.14168)
- [Let's Verify Step by Step (OpenAI, 2023)](https://arxiv.org/abs/2305.20050)
- [Rewarding Progress: Scaling Automated Process Verifiers (2024)](https://arxiv.org/abs/2410.08146)
- [Constitutional AI (Anthropic, 2022)](https://arxiv.org/abs/2212.08073)

### RLHF & RLAIF
- [InstructGPT: Training Language Models to Follow Instructions (OpenAI, 2022)](https://arxiv.org/abs/2203.02155)
- [RLAIF: Reinforcement Learning from AI Feedback (Anthropic, 2023)](https://arxiv.org/abs/2309.00267)

### Blogs & Tutorials
- [Hugging Face: RLHF Tutorial](https://huggingface.co/blog/rlhf)
- [Cameron Wolfe: Reward Models](https://cameronrwolfe.substack.com/p/reward-models)
- [Lilian Weng: RLHF](https://lilianweng.github.io/posts/2023-05-02-rlhf/)

## Next Steps

You now understand the full pipeline from classical RL to modern LLM training. The next step is to implement your own verifier environment and train a small model with it. Start simple (like the Number Guesser, but for text) and gradually increase complexity.
