# utils.py – Utility functions for logging, seeding, device selection, and version checks.
import random
import numpy as np
import torch
import logging
from importlib.metadata import version, PackageNotFoundError


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure root logger and return a module-specific logger.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    return logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """
    Return the best available device: MPS > CUDA > CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_package_version(pkg_name: str) -> str:
    """
    Return the installed version of a package, or 'Not installed' if missing.
    """
    try:
        return version(pkg_name)
    except PackageNotFoundError:
        return "Not installed"


# ── Additional Utilities ─────────────────────────────────────────────────

def check_gpu_availability(logger: logging.Logger = None) -> bool:
    """
    Detect available GPU or MPS device and log status.
    Returns True if CUDA or MPS is available.
    """
    lg = logger or logging.getLogger(__name__)
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        name0 = torch.cuda.get_device_name(0)
        lg.info(f"✅ {count} CUDA GPU(s) detected: {name0}")
        for i in range(1, count):
            lg.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    elif torch.backends.mps.is_available():
        lg.info("✅ MPS device detected")
        return True
    else:
        lg.warning("❌ No GPU detected, using CPU.")
        return False

def get_model_load_kwargs() -> dict:
    """
    Choose device_map and dtype based on available hardware.
    Return keyword arguments for HuggingFace `from_pretrained`.
    """
    device = get_device()
    if device.type in ("cuda", "mps"):
        return {"device_map": "auto", "torch_dtype": torch.float16}
    else:
        return {"device_map": "cpu", "torch_dtype": torch.float32}

def print_package_versions(pkg_names: list, logger: logging.Logger = None) -> None:
    """
    Log installed package versions for specified packages.
    """
    lg = logger or logging.getLogger(__name__)
    # Log installed package versions for specified packages.
    for pkg in pkg_names:
        lg.info(f"{pkg} version: {get_package_version(pkg)}")

