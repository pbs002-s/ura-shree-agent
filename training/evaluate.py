"""
Model Evaluation and Validation Metrics for URA-Shree.
Computes validation loss and token perplexity under torch.no_grad().
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple

from model.model import ShreeTransformerLM


@torch.no_grad()
def evaluate_model(
    model: ShreeTransformerLM,
    dataloader: DataLoader,
    eval_iters: int = 50,
    device: str = "cpu",
    mixed_precision: bool = False,
) -> Tuple[float, float]:
    """
    Evaluates model on validation data.

    Returns:
        mean_loss: Average cross-entropy loss across batches
        perplexity: exp(mean_loss), measuring prediction uncertainty
    """
    model.eval()
    total_loss = 0.0
    actual_iters = 0

    device_type = "cuda" if "cuda" in str(device) else "cpu"
    amp_dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    data_iter = iter(dataloader)

    for _ in range(eval_iters):
        try:
            x, y = next(data_iter)
        except StopIteration:
            # Re-seed iterator if dataloader is smaller than eval_iters
            data_iter = iter(dataloader)
            try:
                x, y = next(data_iter)
            except StopIteration:
                break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if mixed_precision and device_type == "cuda":
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
                _, loss = model(x, targets=y)
        else:
            _, loss = model(x, targets=y)

        if loss is not None:
            total_loss += loss.item()
            actual_iters += 1

    if actual_iters == 0:
        return float("inf"), float("inf")

    mean_loss = total_loss / actual_iters
    perplexity = math.exp(min(mean_loss, 100.0)) # Guard against float overflow

    return mean_loss, perplexity
