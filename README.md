# LLM Summarization with LoRA

A simple framework for fine-tuning and evaluating a seq2seq summarization model (sshleifer/distilbart-cnn-12-6) on CNN/DailyMail using LoRA adapters.

## Setup

### Using Python venv + pip
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Using Conda
```bash
conda create -n llm_env python=3.10 -y
conda activate llm_env
pip install -r requirements.txt
```

## Directory Structure

```
project-root/
├── data.py              # Data loading & preprocessing
├── model.py             # Model setup (seq2seq + LoRA)
├── train.py             # Training loop with logging
├── evaluate.py          # Generation & metrics computation
├── app.py               # Gradio demo interface
├── utils.py             # Utility functions (seed, device, logging)
├── requirements.txt     # Python dependencies
└── outputs/             # Saved adapter weights & tokenizer
```

## Data Preparation

This project uses the CNN/DailyMail summarization dataset via the Hugging Face Datasets library:

- **cnn_dailymail** (`config="3.0.0"`)

Each example in the dataset follows this format:

```text
Article: [full article text]
Summary: [reference summary]
```

## Training

Fine-tune your chosen model with LoRA adapters:

```bash
python train.py 
```

**Logging and checkpoints**  
- Training logs are printed with epoch, global step, and loss.  
- Final adapter and tokenizer are saved to `outputs/<model_name>/`.

## Evaluation

Compute ROUGE and BLEU on your held-out test set:

```bash
python evaluate.py
```

Results will be logged to the console.

## Demo

Launch a simple Gradio interface to interactively test your fine-tuned summarizer:

```bash
python app.py
```

Then visit the local URL printed in your console (e.g., `http://127.0.0.1:7860`).
