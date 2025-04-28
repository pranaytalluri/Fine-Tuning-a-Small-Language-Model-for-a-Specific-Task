from utils import setup_logging, set_seed, get_device
from data import get_dataloaders
from model import build_model
from datasets import load_metric
from transformers import AutoTokenizer
from peft import PeftModel

# ── Configuration ─────────────────────────────────────────────────────────
model_name        = "sshleifer/distilbart-cnn-12-6"
dataset_name      = "cnn_dailymail"
dataset_config    = "3.0.0"
batch_size        = 4
max_input_length  = 256
max_target_length = 64
output_dir        = "/Users/pranaytalluri/Week 9 LLM assignment/outputs"

# ── Setup logging, seed, device ───────────────────────────────────────────
logger = setup_logging()
set_seed()
device = get_device()

def evaluate():
    logger.info(f"Evaluating model from {output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load base model and then apply the saved LoRA adapter
    base_model = build_model(model_name)
    model = PeftModel.from_pretrained(base_model, output_dir).to(device)

    _, val_loader, _ = get_dataloaders(
        model_name,
        dataset_name,
        dataset_config,
        batch_size,
        max_input_length,
        max_target_length
    )

    rouge = load_metric("rouge")
    bleu  = load_metric("bleu")

    model.eval()
    preds, refs = [], []
    for batch in val_loader:
        inputs = batch["input_ids"].to(device)
        attn   = batch["attention_mask"].to(device)
        out    = model.generate(inputs, attention_mask=attn, max_new_tokens=128)

        dec_preds = tokenizer.batch_decode(out, skip_special_tokens=True)
        dec_refs  = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        preds.extend(dec_preds)
        refs.extend(dec_refs)

    rouge_scores = rouge.compute(predictions=preds, references=refs)
    bleu_score   = bleu.compute(
        predictions=[p.split() for p in preds],
        references=[[r.split()] for r in refs]
    )

    logger.info(f"ROUGE: {rouge_scores}")
    logger.info(f"BLEU: {bleu_score}")

if __name__ == "__main__":
    evaluate()