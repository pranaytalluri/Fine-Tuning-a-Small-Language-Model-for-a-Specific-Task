# model.py – Build seq2seq model and apply LoRA adapters for fine-tuning.
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
from utils import setup_logging

logger = setup_logging()

# Load pretrained seq2seq LM and configure LoRA parameter adapters.
def build_model(model_name: str, r: int = 8, alpha: int = 16):
    """
    Load a seq2seq LM and apply LoRA adapters.
    """
    logger.info(f"Loading model: {model_name}")
    # Load base model with automatic device and dtype mapping.
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    # Configure LoRA: adapter rank, scaling factor, and target modules.
    peft_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        inference_mode=False,
    )
    model = get_peft_model(base_model, peft_cfg)
    logger.info(f"Applied LoRA r={r}, alpha={alpha}")
    return model
