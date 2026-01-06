# RL Data Generator

This tool generates synthetic RL datasets for LLM preference optimization using the Gemini API.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your Gemini API key in a `.env` file:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

Run the generator using the CLI:

```bash
python main.py --model gemini-3-flash-preview --type coding,math,summarization,creativity --num_samples 12 --out output_data
```

### Arguments

- `--model`: The Gemini model name to use (e.g., `gemini-3-flash-preview`, `gemini-3-pro-preview`).
- `--type`: Comma-separated list of sample types to generate. Supported: `coding`, `math`, `summarization`, `creativity`.
- `--num_samples`: Total number of samples to generate. They will be distributed evenly across the specified types.
- `--out`: The output directory where `dataset.csv` and `reward_function.py` will be saved.

## Output

- `dataset.csv`: Contains the generated prompts, chosen responses, and rejected responses.
- `reward_function.py`: A Python file containing a basic reward function for the generated data.
