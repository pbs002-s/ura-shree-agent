"""
Coding-Specialized Supervised Fine-Tuning (SFT) for URA-Shree.
Fine-tunes the base model on agent tool invocation syntax, code generation, and test execution traces.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from typing import Optional

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.config import ProjectConfig, ModelConfig
from model.model import ShreeTransformerLM
from data.dataset import CausalLMDataset, create_dataloader
from training.evaluate import evaluate_model
from training.checkpoint import CheckpointManager
from training.train import configure_optimizers, get_lr_cosine_warmup


def train_coding_sft(
    config_path: str = "configs/small.yaml",
    base_checkpoint: str = "checkpoints/best.pt",
    output_checkpoint: str = "checkpoints/coding_best.pt",
    steps: int = 30,
    learning_rate: float = 2.0e-4,
    batch_size: int = 8,
) -> float:
    """
    Executes specialized coding fine-tuning.
    """
    print("=" * 70)
    print("      URA-Shree: Coding & Agent Tool Fine-Tuning (SFT)")
    print("=" * 70)

    config = ProjectConfig.load_from_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using compute target: {device}")

    # 1. Initialize model and load base weights
    model = ShreeTransformerLM(config.model).to(device)

    if os.path.exists(base_checkpoint):
        print(f"[Model] Loading base pre-trained weights from: {base_checkpoint}")
        raw_ckpt = torch.load(base_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(raw_ckpt["model_state_dict"])
    else:
        print(f"[Warning] Base checkpoint {base_checkpoint} not found. Training from scratch.")

    # 2. Check or prepare coding datasets
    train_bin = os.path.join("datasets", "coding_train.bin")
    val_bin = os.path.join("datasets", "coding_val.bin")

    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        from data.prepare_coding_dataset import prepare_coding_dataset
        prepare_coding_dataset()

    # Context window for SFT
    seq_len = min(512, config.model.max_seq_len)
    train_ds = CausalLMDataset(train_bin, seq_len=seq_len)
    val_ds = CausalLMDataset(val_bin, seq_len=seq_len)

    train_loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"[Dataset] Coding train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # 3. Configure Optimizer & Mixed Precision
    optimizer = configure_optimizers(
        model,
        learning_rate=learning_rate,
        weight_decay=0.05,
        device_type=device.type,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    # 4. SFT Loop
    model.train()
    data_iter = iter(train_loader)
    best_val_loss = float("inf")
    warmup_steps = 5

    print("\n" + "-" * 75)
    print(f"{'Step':<8} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val PPL':<9} | {'LR':<10} | {'VRAM':<8}")
    print("-" * 75, flush=True)

    for step in range(1, steps + 1):
        lr = get_lr_cosine_warmup(
            step=step,
            warmup_steps=warmup_steps,
            max_steps=steps,
            peak_lr=learning_rate,
            min_lr=learning_rate * 0.1,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                _, loss = model(x, targets=y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Log every 5 steps or final step
        if step % 5 == 0 or step == steps:
            val_loss, val_ppl = evaluate_model(
                model,
                val_loader,
                eval_iters=10,
                device=str(device),
                mixed_precision=use_amp,
            )
            vram_mb = torch.cuda.memory_allocated(0) / (1024 ** 2) if device.type == "cuda" else 0.0
            print(
                f"{step:<8} | {loss.item():<12.4f} | {val_loss:<10.4f} | {val_ppl:<9.2f} | {lr:<10.2e} | {vram_mb:.1f} MB",
                flush=True,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
                torch.save(
                    {
                        "step": step,
                        "val_loss": val_loss,
                        "model_state_dict": model.state_dict(),
                        "config": {"model": config.model.to_dict()},
                    },
                    output_checkpoint,
                )
            model.train()

    print("-" * 75)
    print(f"[SFT Complete] Coding-specialized model saved to: {output_checkpoint}")
    print(f"[SFT Complete] Best Validation Loss: {best_val_loss:.4f}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Fine-tune URA-Shree on coding agent tasks.")
    parser.add_argument("--config", type=str, default="configs/small.yaml")
    parser.add_argument("--base", type=str, default="checkpoints/best.pt")
    parser.add_argument("--output", type=str, default="checkpoints/coding_best.pt")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    args = parser.parse_args()

    train_coding_sft(
        config_path=args.config,
        base_checkpoint=args.base,
        output_checkpoint=args.output,
        steps=args.steps,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
