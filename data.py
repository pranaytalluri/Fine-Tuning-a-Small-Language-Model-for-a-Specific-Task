# data.py – Load, preprocess, and prepare summarization datasets for PyTorch.
# data.py

import logging
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from utils import setup_logging

logger = setup_logging()

# PyTorch Dataset wrapper converting tokenized inputs into tensors.
class SummarizationDataset(Dataset):
    """Wrap a Hugging Face Dataset for PyTorch DataLoader, converting features to torch.Tensor."""
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }

def get_dataloaders(
    model_name: str,
    dataset_name: str = "cnn_dailymail",
    dataset_config: str = "3.0.0",
    batch_size: int = 2,
    max_input_length: int = 512,
    max_target_length: int = 64,
):
    """
    Load and preprocess summarization datasets via Hugging Face Datasets.
    Returns train, val, test DataLoaders.
    """
    # Load the CNN/DailyMail dataset via Hugging Face Datasets.
    logger.info(f"Loading dataset: {dataset_name}/{dataset_config}")
    ds = load_dataset(dataset_name, dataset_config)

    # Subsample each split (3%) to limit memory usage.
    frac = 0.03
    train_n = max(1, int(len(ds["train"]) * frac))
    val_n   = max(1, int(len(ds["validation"]) * frac))
    test_n  = max(1, int(len(ds["test"]) * frac))
    ds["train"]      = ds["train"].shuffle(seed=42).select(range(train_n))
    ds["validation"] = ds["validation"].shuffle(seed=42).select(range(val_n))
    ds["test"]       = ds["test"].shuffle(seed=42).select(range(test_n))

    # Identify source text and target summary columns.
    input_key  = "article"
    target_key = "highlights" if "highlights" in ds["train"].column_names else "summary"

    # Initialize tokenizer for the chosen model.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Preprocessing: prefix articles and tokenize inputs and summaries.
    def preprocess(batch):
        # Add explicit prefix
        texts   = ["Article: " + t for t in batch[input_key]]
        targets = batch[target_key]

        # Tokenize inputs
        enc_inputs = tokenizer(
            texts,
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
        )
        # Tokenize targets
        with tokenizer.as_target_tokenizer():
            enc_targets = tokenizer(
                targets,
                max_length=max_target_length,
                padding="max_length",
                truncation=True,
            )

        enc_inputs["labels"] = enc_targets["input_ids"]
        return enc_inputs

    # Apply preprocessing and drop raw columns.
    train_hf = ds["train"].map(
        preprocess,
        batched=True,
        remove_columns=ds["train"].column_names,
    )
    val_hf   = ds["validation"].map(
        preprocess,
        batched=True,
        remove_columns=ds["validation"].column_names,
    )
    test_hf  = ds["test"].map(
        preprocess,
        batched=True,
        remove_columns=ds["test"].column_names,
    )

    # Wrap tokenized data into PyTorch Dataset objects.
    train_ds = SummarizationDataset(train_hf)
    val_ds   = SummarizationDataset(val_hf)
    test_ds  = SummarizationDataset(test_hf)

    # Create DataLoader for batching during training and evaluation.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    logger.info(
        f"Loaded {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test examples\n"
        f"batch_size={batch_size}, max_input_length={max_input_length}, max_target_length={max_target_length}"
    )

    return train_loader, val_loader, test_loader
