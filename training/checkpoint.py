"""
Atomic Checkpointing Subsystem for URA-Shree.
Saves and restores model weights, optimizer states, GradScaler states,
and training step counters with corruption protection and best-model tracking.
"""

import os
import glob
import shutil
import torch
from typing import Dict, Any, Optional, Tuple

from model.config import ProjectConfig, ModelConfig
from model.model import ShreeTransformerLM


class CheckpointManager:
    """
    Manages atomic checkpoint saving, pruning of old checkpoints,
    and tracking of the best performing model based on validation loss.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints", keep_last: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last = keep_last
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(
        self,
        model: ShreeTransformerLM,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler],
        step: int,
        val_loss: float,
        config: ProjectConfig,
        is_best: bool = False,
    ) -> str:
        """
        Saves training state atomically to disk.
        Writes to a temporary file first, then renames to avoid partial-write corruption.
        """
        checkpoint_data = {
            "step": step,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "config": {
                "model": config.model.to_dict(),
                "training": config.training.to_dict(),
                "checkpoint": config.checkpoint.to_dict(),
            },
        }

        # Filename for this specific step
        step_filename = f"checkpoint_step_{step:07d}.pt"
        final_path = os.path.join(self.checkpoint_dir, step_filename)
        temp_path = final_path + ".tmp"

        # 1. Save to temporary path
        torch.save(checkpoint_data, temp_path)

        # 2. Atomic rename
        if os.path.exists(final_path):
            os.remove(final_path)
        os.replace(temp_path, final_path)

        # 3. Always update last.pt symlink/copy
        last_path = os.path.join(self.checkpoint_dir, "last.pt")
        shutil.copyfile(final_path, last_path)

        # 4. If this is the best validation loss so far, update best.pt
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            shutil.copyfile(final_path, best_path)
            print(f"[Checkpoint] New best model saved with validation loss: {val_loss:.4f} -> {best_path}")

        # 5. Prune old step checkpoints beyond keep_last
        self._prune_old_checkpoints()

        return final_path

    def _prune_old_checkpoints(self) -> None:
        """Keeps only the most recent keep_last step checkpoints."""
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_step_*.pt")
        checkpoints = sorted(glob.glob(pattern))

        if len(checkpoints) > self.keep_last:
            to_remove = checkpoints[: len(checkpoints) - self.keep_last]
            for ckpt in to_remove:
                try:
                    os.remove(ckpt)
                except OSError:
                    pass

    @staticmethod
    def load(
        checkpoint_path: str,
        model: ShreeTransformerLM,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        device: str = "cpu",
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        Loads a saved checkpoint and restores model and optimizer states.

        Returns:
            step: Restored step number
            val_loss: Validation loss at checkpoint time
            config_dict: Saved configuration dictionary
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        print(f"[Checkpoint] Loading checkpoint from: {checkpoint_path} (device: {device})")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Restore model weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scaler state if provided
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        step = checkpoint.get("step", 0)
        val_loss = checkpoint.get("val_loss", float("inf"))
        config_dict = checkpoint.get("config", {})

        print(f"[Checkpoint] Restored step: {step} | Val Loss: {val_loss:.4f}")
        return step, val_loss, config_dict
