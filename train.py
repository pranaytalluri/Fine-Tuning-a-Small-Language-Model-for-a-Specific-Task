# train.py – Fine-tune LoRA adapters with training loop, scheduler, and checkpointing.
import os
from utils import setup_logging, set_seed, get_device, get_package_version
from data import get_dataloaders
from model import build_model
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from peft import PeftModel

# ── Configuration: dataset, model, hyperparameters, and output directory.
model_name         = "sshleifer/distilbart-cnn-12-6"
dataset_name       = "cnn_dailymail"
dataset_config     = "3.0.0"
output_dir         = "/Users/pranaytalluri/Week 9 LLM assignment/outputs"
num_epochs         = 10
batch_size         = 4
learning_rate      = 1e-4
grad_accum         = 8
max_input_length   = 512
max_target_length  = 64

# ── Setup logging, seed, device, versions ────────────────────────────────
logger = setup_logging()
set_seed()
device = get_device()

# Log software versions and device details for reproducibility.
logger.info(f"PyTorch version: {get_package_version('torch')}")
logger.info(f"Transformers version: {get_package_version('transformers')}")
logger.info(f"PEFT version: {get_package_version('peft')}")
logger.info(f"Datasets version: {get_package_version('datasets')}")

# Main training loop: forward, backward, gradient accumulation, and checkpoint saving.
def train(model, train_loader, val_loader, device, output_dir):
    """Training loop with LoRA adapter saving and logging."""
    logger.info("Starting training")
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs // grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    model.train()
    optimizer.zero_grad()
    epoch_loss = 0.0
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            # Forward pass: compute loss on current batch.
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()
            epoch_loss += loss.item() * grad_accum

            # Perform optimizer update and scheduler step after gradient accumulation.
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step = epoch * len(train_loader) + step + 1
                logger.info(f"Global step {global_step}: loss {loss.item() * grad_accum:.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")
        # reset epoch_loss for next epoch
        epoch_loss = 0.0

        # Save model state for this epoch under output directory.
        ckpt = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
        model.save_pretrained(ckpt)
        logger.info(f"Saved checkpoint to {ckpt}")

    # After training, save LoRA adapter weights and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info(f"Saving final PEFT adapter to {output_dir}")
    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Adapter weights and tokenizer saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error saving adapter: {e}")
        raise RuntimeError(f"Failed to save adapter: {e}")

if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    train_loader, val_loader, _ = get_dataloaders(
        model_name,
        dataset_name,
        dataset_config,
        batch_size,
        max_input_length,
        max_target_length
    )
    model = build_model(model_name).to(device)
    train(model, train_loader, val_loader, device, output_dir)
