# app.py

import gradio as gr
import torch
from transformers import AutoTokenizer
from peft import PeftModel
from utils import get_device, get_model_load_kwargs
from model import build_model

# ── Configuration ─────────────────────────────────────────────────────────
model_name    = "sshleifer/distilbart-cnn-12-6"
adapter_dir   = "/Users/pranaytalluri/Week 9 LLM assignment/outputs"       # where you saved your LoRA adapter
max_input_len = 512
max_new_tokens= 64

# ── Device & model loading ────────────────────────────────────────────────
device = get_device()

# 1) Load base seq2seq + LoRA adapter
base_model = build_model(model_name)
model = PeftModel.from_pretrained(
    base_model,
    adapter_dir,
    is_trainable=False,
    local_files_only=True  # 🛠️ this is the key
).to(device)
model.config.use_cache = True
model.eval()

# 2) Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ── Inference function ────────────────────────────────────────────────────
def summarize(article: str) -> str:
    """
    Take a raw article string, prepend the "Article:" prefix,
    tokenize, generate, and decode a summary.
    """
    prompt = "Article: " + article.strip()
    inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # ⬅️ reduce input length
        ).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False
        )
    summary = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    # If the model echoes the prompt, strip it off:
    return summary.replace(prompt, "").strip()

# ── Gradio Interface ──────────────────────────────────────────────────────
iface = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=6, placeholder="Paste your article here...", label="Article"),
    outputs=gr.Textbox(lines=4, label="Summary"),
    title="LoRA-tuned BART Summarizer",
    description=(
        "This demo uses a distilbart-cnn model fine-tuned via LoRA adapters "
        "on CNN/DailyMail. Enter any news-style article and get a concise summary."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    # Launch in local server; you can add share=True to get a public link
    iface.launch(share=False)