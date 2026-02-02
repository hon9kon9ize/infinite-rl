# LLM-as-a-Judge Reward Function

## Overview

The **LLMJudgeRewardFunction** uses a remote LLM-based reward model to evaluate the quality of model responses. It leverages the [Skywork Reward Model (V2-Qwen3-4B)](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-4B) via an sglang API endpoint to provide continuous quality scores.

## Features

- **Remote Scoring**: Calls a remote sglang server for LLM-based evaluation
- **Flexible Normalization**: Supports both raw scores and normalized [0, 1] range
- **Threshold Control**: Can filter scores below a configurable threshold
- **Chat Template Support**: Automatically formats conversations using the model's tokenizer
- **Error Handling**: Graceful fallback when API is unavailable
- **Efficient Batching**: Supports batch scoring (when extended to curriculum integration)

## Architecture

### Setup Requirements

1. **sglang Server**: Run the Skywork Reward Model on a sglang server
   ```bash
   python -m sglang.launch_server \
       --model-path Skywork/Skywork-Reward-V2-Qwen3-4B \
       --mem-fraction-static 0.9 \
       --tp 1 \
       --device cuda \
       --dtype bfloat16 \
       --host 0.0.0.0 \
       --port 8000 \
       --context-length 16384 \
       --is-embedding
   ```

2. **Python Dependencies**:
   ```bash
   pip install requests transformers
   pip install sglang  # For running the server
   ```

### API Endpoint

The function communicates with the `/classify` endpoint:
- **URL**: `http://{api_host}:{api_port}/classify`
- **Method**: POST
- **Request Payload**:
  ```json
  {
    "model": "Skywork/Skywork-Reward-V2-Qwen3-4B",
    "text": ["formatted_conversation_1", "formatted_conversation_2", ...]
  }
  ```
- **Response**:
  ```json
  [
    {"embedding": [23.125]},
    {"embedding": [3.578125]},
    ...
  ]
  ```

## Usage

### Basic Setup

```python
from infinite_rl.reward_functions import LLMJudgeRewardFunction
from infinite_rl.task import Task

# Create reward function pointing to your sglang server
judge = LLMJudgeRewardFunction(
    api_host="172.16.17.29",
    api_port=8000,
    model_name="Skywork/Skywork-Reward-V2-Qwen3-4B",
    normalize=True,  # Normalize to [0, 1]
)

# Initialize (loads tokenizer)
judge.initialize()

# Create a task with model output
task = Task(
    task_id="q1",
    task_name="Math Problem",
    task_type="math",
    level=0,
    prompt="What is 2+2?",
    expected_answer="4"
)
task.model_output = """<think>Let me add 2+2</think>
<answer>4</answer>"""

# Compute reward
reward = judge.compute_reward(task)
print(f"Score: {reward.score:.4f}")
print(f"Info: {reward.info}")
```

### Configuration Options

```python
judge = LLMJudgeRewardFunction(
    # API Configuration
    api_host="localhost",           # sglang server host
    api_port=8000,                  # sglang server port
    model_name="Skywork/Skywork-Reward-V2-Qwen3-4B",
    
    # Scoring Configuration
    score_threshold=-100.0,         # Minimum acceptable score
    normalize=True,                 # Normalize to [0, 1]?
    
    # Function Configuration
    task_name="llm_judge",          # Internal name
    timeout=30,                     # API timeout in seconds
    answer_tag="answer",            # Tag for answer extraction
    think_tag="think",              # Tag for reasoning extraction
)
```

## Score Interpretation

### Raw Scores
The Skywork Reward Model outputs continuous scores that typically range from negative to positive values:
- Example scores: 23.125 (excellent), 3.578 (poor), -5.0 (very poor)
- Interpretation depends on the model's training

### Normalized Scores
When `normalize=True`, raw scores are converted using tanh-based normalization:

$$\text{normalized} = \frac{\tanh(\text{raw\_score} / 10) + 1}{2}$$

This maps the score to [0, 1] while preserving relative ordering:
- **0.0-0.25**: Very poor responses
- **0.25-0.5**: Below average responses
- **0.5-0.75**: Good responses
- **0.75-1.0**: Excellent responses

### Threshold Filtering
Scores below `score_threshold` return 0.0:
```python
judge = LLMJudgeRewardFunction(score_threshold=-50.0)
# Scores < -50 will be returned as 0.0
```

## Integration with Curriculum Learning

The LLM judge can be integrated as an **auxiliary reward function** in curriculum learning:

```python
from infinite_rl.curriculum import CurriculumLearning
from infinite_rl.reward_functions import LLMJudgeRewardFunction

curriculum = CurriculumLearning(
    aux_weight=0.2,  # 20% from auxiliary rewards
    aux_reward_config={
        "llm_judge": {
            "enabled": True,
            "api_host": "172.16.17.29",
            "api_port": 8000,
            "normalize": True,
        }
    }
)
```

## Error Handling

The function gracefully handles various error scenarios:

| Error | Behavior |
|-------|----------|
| Empty prompt | Returns 0.0 with info message |
| Empty model output | Returns 0.0 with info message |
| API timeout | Returns 0.0 with info message |
| Connection error | Returns 0.0 with info message |
| Malformed API response | Returns 0.0 with info message |
| Score below threshold | Returns 0.0 with threshold info |
| Tokenizer not available | Raises RuntimeError on initialize() |

## Example Scenarios

### Math Problem Evaluation
```python
prompt = "Jane has 12 apples. She gives 4 to Mark, buys 1 more, " \
         "and splits equally with her 2 siblings. How many each?"

response_good = """<think>
1. Jane starts: 12 apples
2. Give 4 to Mark: 12 - 4 = 8
3. Buy 1 more: 8 + 1 = 9
4. Split 3 ways: 9 ÷ 3 = 3 each
</think>
<answer>Each person gets 3 apples</answer>"""

response_bad = """<think>
Jane has apples and shares them
</think>
<answer>4.5 apples each</answer>"""

task_good = Task(...)
task_good.model_output = response_good
reward_good = judge.compute_reward(task_good)
# Expected: High score (e.g., 0.9+)

task_bad = Task(...)
task_bad.model_output = response_bad
reward_bad = judge.compute_reward(task_bad)
# Expected: Low score (e.g., 0.2-)
```

### Batch Processing
```python
tasks = [...]  # List of Task objects with model_output

rewards = []
for task in tasks:
    reward = judge.compute_reward(task)
    rewards.append(reward.score)

avg_score = sum(rewards) / len(rewards)
print(f"Average quality: {avg_score:.4f}")
```

## Testing

Run the comprehensive test suite:
```bash
python -m pytest tests/test_llm_judge_reward_function.py -v
```

Test coverage includes:
- ✅ Initialization and error handling
- ✅ Chat template formatting
- ✅ API communication and error scenarios
- ✅ Score normalization
- ✅ Threshold filtering
- ✅ End-to-end reward computation
- ✅ Integration scenarios

## Performance Considerations

1. **API Latency**: Each call requires network request to sglang server (~50-200ms)
2. **Batch Processing**: Consider batching multiple responses in a single API call for efficiency
3. **Timeout**: Default 30 seconds; adjust based on your server's performance
4. **GPU Memory**: Ensure sglang server has sufficient GPU memory (bfloat16 mode helps)

## Advanced Configuration

### Custom Normalization

To use custom score normalization instead of tanh:
```python
class CustomJudge(LLMJudgeRewardFunction):
    def _normalize_score(self, raw_score: float) -> float:
        # Custom normalization logic
        return (raw_score - min_val) / (max_val - min_val)
```

### Batch Processing

When integrated with curriculum learning, batch the API calls:
```python
# Process multiple tasks at once (requires curriculum integration)
# See curriculum.py for implementation details
```

## References

- **Skywork Reward Model**: [Hugging Face Model Card](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-4B)
- **sglang Documentation**: [GitHub Repository](https://github.com/hpcaitech/sglang)
- **LLM Reward Models**: [Open-source evaluation paper](https://arxiv.org/abs/2312.03122)

## Troubleshooting

### "Failed to initialize LLMJudgeRewardFunction"
- Ensure transformers is installed: `pip install transformers`
- Ensure model is accessible from HuggingFace or locally cached

### "Connection refused" or "Connection timeout"
- Verify sglang server is running
- Check api_host and api_port are correct
- Ensure network connectivity to sglang server

### Scores all return 0.0
- Check sglang server logs for errors
- Verify API response format matches expected structure
- Try disabling score threshold to see raw values

### High variability in scores
- This may be expected behavior of the reward model
- Consider using running averages in curriculum learning
- Check model's training data and intended use cases
