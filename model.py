from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
from utils import setup_logging

logger = setup_logging()

def build_model(model_name: str, r: int = 8, alpha: int = 16):
    """
    Load a seq2seq LM and apply LoRA adapters.
    """
    logger.info(f"Loading model: {model_name}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    peft_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        inference_mode=False,
    )
    model = get_peft_model(base_model, peft_cfg)
    logger.info(f"Applied LoRA r={r}, alpha={alpha}")
    return model