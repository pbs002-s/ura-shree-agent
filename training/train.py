"""
High-Performance Training Pipeline for URA-Shree.
Trains custom decoder-only Transformer with mixed precision (FP16/BF16),
gradient accumulation, AdamW optimizer, cosine warmup scheduler, and live telemetry.
"""

import os
import sys
import time
import math
import argparse
from typing import Optional, Dict, Any, Tuple, List
import torch
import torch.nn as nn
from torch.optim import AdamW

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.config import ProjectConfig, ModelConfig, TrainingConfig
from model.model import ShreeTransformerLM
from data.dataset import CausalLMDataset, create_dataloader
from training.evaluate import evaluate_model
from training.checkpoint import CheckpointManager


def configure_optimizers(
    model: ShreeTransformerLM,
    learning_rate: float,
    weight_decay: float,
    device_type: str,
) -> AdamW:
    """
    Configures AdamW with weight decay applied ONLY to 2D weight matrices.
    Biases and 1D normalization gain parameters (RMSNorm, LayerNorm) receive 0 weight decay.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # If parameter has 2 or more dimensions, apply weight decay
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Use fused AdamW kernel on CUDA if available for maximum speed
    use_fused = (device_type == "cuda") and ("fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
    optimizer = AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer


def get_lr_cosine_warmup(
    step: int,
    warmup_steps: int,
    max_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """
    Computes current learning rate using linear warmup followed by cosine annealing decay.
    """
    # 1. Linear warmup phase
    if step < warmup_steps:
        return peak_lr * (step + 1) / (warmup_steps + 1)

    # 2. Beyond max steps, maintain min_lr
    if step > max_steps:
        return min_lr

    # 3. Cosine decay phase
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)


def train(
    config_path: str = "configs/small.yaml",
    resume_path: Optional[str] = None,
    override_max_steps: Optional[int] = None,
) -> None:
    """
    Main training execution function.
    """
    print("=" * 70)
    print("           URA-Shree: Neural Model Training Pipeline")
    print("=" * 70)

    # 1. Load and validate configuration
    config = ProjectConfig.load_from_yaml(config_path)
    train_cfg = config.training
    model_cfg = config.model

    if override_max_steps is not None:
        train_cfg.max_steps = override_max_steps

    # 2. Hardware Detection
    device_str = train_cfg.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(f"[Device] Using compute target: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Device] GPU: {gpu_name} ({vram_total:.2f} GB VRAM)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 3. Initialize Model
    torch.manual_seed(train_cfg.seed)
    model = ShreeTransformerLM(model_cfg).to(device)

    # 4. Configure Optimizer & Mixed Precision Scaler
    optimizer = configure_optimizers(
        model,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        device_type=device.type,
    )

    use_amp = train_cfg.mixed_precision and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    print(f"[Training] Mixed Precision Enabled: {use_amp} (dtype: {amp_dtype})")
    print(f"[Training] Micro Batch Size: {train_cfg.batch_size} | Gradient Accumulation Steps: {train_cfg.grad_accum_steps}")
    print(f"[Training] Effective Batch Size: {train_cfg.batch_size * train_cfg.grad_accum_steps} tokens * {model_cfg.max_seq_len}")

    # 5. Checkpoint Manager & Resumption
    ckpt_manager = CheckpointManager(checkpoint_dir=config.checkpoint.dir, keep_last=config.checkpoint.keep_last)
    start_step = 0
    best_val_loss = float("inf")

    if resume_path:
        start_step, best_val_loss, _ = ckpt_manager.load(
            resume_path, model, optimizer=optimizer, scaler=scaler, device=str(device)
        )
        print(f"[Resume] Successfully resumed from step {start_step} with best val loss: {best_val_loss:.4f}")

    # 6. Load Datasets
    train_bin = os.path.join("datasets", "train.bin")
    val_bin = os.path.join("datasets", "val.bin")

    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        print(f"[Error] Datasets not found at {train_bin}. Running data/prepare_dataset.py first...")
        from data.prepare_dataset import prepare_dataset
        prepare_dataset(output_dir="datasets")

    train_dataset = CausalLMDataset(train_bin, seq_len=model_cfg.max_seq_len)
    val_dataset = CausalLMDataset(val_bin, seq_len=model_cfg.max_seq_len)

    train_loader = create_dataloader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False)

    print(f"[Dataset] Train samples: {len(train_dataset):,} | Val samples: {len(val_dataset):,}")

    # 7. Training Loop
    model.train()
    data_iter = iter(train_loader)
    step = start_step

    t_start = time.time()
    accum_loss = 0.0

    print("\n" + "-" * 82)
    print(f"{'Step':<8} | {'Train Loss':<11} | {'Val Loss':<10} | {'Val PPL':<9} | {'LR':<10} | {'Tok/Sec':<9} | {'VRAM':<8}")
    print("-" * 82, flush=True)

    while step < train_cfg.max_steps:
        # Determine learning rate for this step
        lr = get_lr_cosine_warmup(
            step,
            warmup_steps=train_cfg.warmup_steps,
            max_steps=train_cfg.max_steps,
            peak_lr=train_cfg.learning_rate,
            min_lr=train_cfg.min_learning_rate,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Zero gradients at start of accumulation window
        optimizer.zero_grad(set_to_none=True)
        micro_loss_sum = 0.0

        for micro_step in range(train_cfg.grad_accum_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Forward pass with mixed precision
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    _, loss = model(x, targets=y)
                    # Normalize loss for gradient accumulation
                    loss = loss / train_cfg.grad_accum_steps
                scaler.scale(loss).backward()
            else:
                _, loss = model(x, targets=y)
                loss = loss / train_cfg.grad_accum_steps
                loss.backward()

            micro_loss_sum += loss.item() * train_cfg.grad_accum_steps

        # Gradient clipping
        if use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.grad_clip)

        # Optimizer step
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        step += 1
        accum_loss += micro_loss_sum

        # Regular training step telemetry (every 5 steps or end)
        log_interval = min(5, train_cfg.eval_interval)
        if step % log_interval == 0 or step == train_cfg.max_steps:
            elapsed = time.time() - t_start
            tokens_processed = (
                log_interval
                * train_cfg.batch_size
                * train_cfg.grad_accum_steps
                * model_cfg.max_seq_len
            )
            tokens_per_sec = tokens_processed / max(0.001, elapsed)
            avg_train_loss = accum_loss / log_interval

            # Check if this is an evaluation step
            is_eval = (step % train_cfg.eval_interval == 0) or (step == train_cfg.max_steps)
            if is_eval:
                val_loss, val_ppl = evaluate_model(
                    model,
                    val_loader,
                    eval_iters=train_cfg.eval_iters,
                    device=str(device),
                    mixed_precision=use_amp,
                )
                val_loss_str = f"{val_loss:.4f}"
                val_ppl_str = f"{val_ppl:.2f}"
            else:
                val_loss_str = "..."
                val_ppl_str = "..."

            vram_mb = torch.cuda.memory_allocated(0) / (1024 ** 2) if device.type == "cuda" else 0.0
            print(
                f"{step:<8} | {avg_train_loss:<11.4f} | {val_loss_str:<10} | {val_ppl_str:<9} | {lr:<10.2e} | {tokens_per_sec:<9.0f} | {vram_mb:.1f} MB",
                flush=True,
            )

            if is_eval:
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss

                if step % train_cfg.save_interval == 0 or is_best or step == train_cfg.max_steps:
                    ckpt_manager.save(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler if use_amp else None,
                        step=step,
                        val_loss=val_loss,
                        config=config,
                        is_best=is_best,
                    )

            accum_loss = 0.0
            t_start = time.time()
            model.train()

    print("-" * 70)
    print(f"[Training Complete] Final Best Validation Loss: {best_val_loss:.4f}")
    print(f"[Checkpoints] Saved to directory: {config.checkpoint.dir}")


def main():
    parser = argparse.ArgumentParser(description="Train URA-Shree Transformer LM from scratch.")
    parser.add_argument("--config", type=str, default="configs/small.yaml", help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--max-steps", type=int, default=None, help="Override total steps")
    args = parser.parse_args()

    train(config_path=args.config, resume_path=args.resume, override_max_steps=args.max_steps)


if __name__ == "__main__":
    main()
