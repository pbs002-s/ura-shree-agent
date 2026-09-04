"""
Unit Tests for URA-Shree Training Pipeline, Schedulers, and Checkpoints.
Verifies learning rate scheduling, optimizer parameter groupings,
atomic checkpoint restoration, and convergence on toy batches.
"""

import os
import sys
import tempfile
import math
import torch
import pytest

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.config import ModelConfig, TrainingConfig, CheckpointConfig, ProjectConfig
from model.model import ShreeTransformerLM
from data.dataset import CausalLMDataset, create_dataloader
from training.checkpoint import CheckpointManager
from training.evaluate import evaluate_model
from training.train import configure_optimizers, get_lr_cosine_warmup


@pytest.fixture
def tiny_project():
    """Returns a tiny ModelConfig and ProjectConfig for testing."""
    m_cfg = ModelConfig(
        name="test-training-tiny",
        vocab_size=128,
        max_seq_len=32,
        embed_dim=32,
        num_layers=2,
        num_heads=2,
        intermediate_dim=64,
        dropout=0.0,
        tie_weights=True,
    )
    t_cfg = TrainingConfig(
        batch_size=4,
        grad_accum_steps=1,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_steps=10,
        max_steps=100,
        device="cpu",
    )
    c_cfg = CheckpointConfig(dir="checkpoints_test", keep_last=2)
    return ProjectConfig(model=m_cfg, training=t_cfg, checkpoint=c_cfg)


def test_cosine_warmup_schedule():
    """Verify linear warmup and cosine decay boundary properties."""
    peak = 6e-4
    min_lr = 6e-5
    warmup = 100
    max_steps = 1000

    # At step 0, lr should be small
    lr_0 = get_lr_cosine_warmup(0, warmup, max_steps, peak, min_lr)
    assert 0.0 < lr_0 < peak

    # Near end of warmup, lr should approach peak
    lr_warmup = get_lr_cosine_warmup(warmup - 1, warmup, max_steps, peak, min_lr)
    assert math.isclose(lr_warmup, peak, rel_tol=0.05)

    # At max steps, lr should be min_lr
    lr_end = get_lr_cosine_warmup(max_steps, warmup, max_steps, peak, min_lr)
    assert math.isclose(lr_end, min_lr, rel_tol=1e-5)

    # Monotonic decrease after warmup
    lr_mid1 = get_lr_cosine_warmup(300, warmup, max_steps, peak, min_lr)
    lr_mid2 = get_lr_cosine_warmup(600, warmup, max_steps, peak, min_lr)
    assert lr_mid1 > lr_mid2


def test_optimizer_parameter_groups(tiny_project):
    """Verify that weight decay is decoupled (2D weights get decay, 1D/biases get 0)."""
    model = ShreeTransformerLM(tiny_project.model)
    optimizer = configure_optimizers(model, learning_rate=1e-3, weight_decay=0.1, device_type="cpu")

    assert len(optimizer.param_groups) == 2
    decay_group = optimizer.param_groups[0]
    no_decay_group = optimizer.param_groups[1]

    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0

    # Ensure all decay params are at least 2D
    for p in decay_group["params"]:
        assert p.dim() >= 2

    # Ensure all no-decay params are 1D (e.g. RMSNorm scales or biases)
    for p in no_decay_group["params"]:
        assert p.dim() < 2


def test_evaluation_loop(tiny_project):
    """Verify evaluate_model computes positive loss and perplexity >= 1.0."""
    model = ShreeTransformerLM(tiny_project.model)
    tokens = torch.randint(0, tiny_project.model.vocab_size, (200,))
    dataset = CausalLMDataset(tokens, seq_len=tiny_project.model.max_seq_len)
    loader = create_dataloader(dataset, batch_size=2, shuffle=False)

    loss, ppl = evaluate_model(model, loader, eval_iters=5, device="cpu")
    assert loss > 0.0
    assert ppl >= 1.0


def test_checkpoint_atomic_save_and_load(tiny_project):
    """Verify checkpoint save and restoration recovers exact weights and step."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiny_project.checkpoint.dir = tmp_dir
        ckpt_manager = CheckpointManager(checkpoint_dir=tmp_dir, keep_last=2)

        model = ShreeTransformerLM(tiny_project.model)
        optimizer = configure_optimizers(model, learning_rate=1e-3, weight_decay=0.1, device_type="cpu")

        # Record weight at step 42
        step = 42
        val_loss = 2.345
        save_path = ckpt_manager.save(
            model=model,
            optimizer=optimizer,
            scaler=None,
            step=step,
            val_loss=val_loss,
            config=tiny_project,
            is_best=True,
        )
        assert os.path.exists(save_path)
        assert os.path.exists(os.path.join(tmp_dir, "best.pt"))
        assert os.path.exists(os.path.join(tmp_dir, "last.pt"))

        # Mutate model weights
        orig_weight = model.embedding.token_embeddings.weight.clone()
        with torch.no_grad():
            model.embedding.token_embeddings.weight.add_(1.0)

        assert not torch.equal(model.embedding.token_embeddings.weight, orig_weight)

        # Restore from checkpoint
        new_model = ShreeTransformerLM(tiny_project.model)
        restored_step, restored_loss, _ = ckpt_manager.load(
            save_path, new_model, optimizer=None, scaler=None, device="cpu"
        )

        assert restored_step == step
        assert math.isclose(restored_loss, val_loss, rel_tol=1e-4)
        assert torch.equal(new_model.embedding.token_embeddings.weight, orig_weight)


def test_overfit_toy_batch(tiny_project):
    """
    CRITICAL ML TEST:
    Verify that backpropagation, loss computation, and AdamW gradient updates
    actively learn and reduce cross-entropy loss on a toy batch.
    """
    model = ShreeTransformerLM(tiny_project.model)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    # Fixed toy batch
    x = torch.randint(0, tiny_project.model.vocab_size, (2, tiny_project.model.max_seq_len))
    y = torch.randint(0, tiny_project.model.vocab_size, (2, tiny_project.model.max_seq_len))

    # Initial loss
    _, initial_loss = model(x, targets=y)
    init_loss_val = initial_loss.item()

    # Train for 25 steps on this batch
    for _ in range(25):
        optimizer.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        optimizer.step()

    _, final_loss = model(x, targets=y)
    final_loss_val = final_loss.item()

    # Loss must have dropped significantly
    assert final_loss_val < init_loss_val, (
        f"Model failed to learn! Initial loss: {init_loss_val:.4f}, Final loss: {final_loss_val:.4f}"
    )
    assert final_loss_val < init_loss_val * 0.5, (
        f"Expected at least 50% loss drop. Init: {init_loss_val:.4f}, Final: {final_loss_val:.4f}"
    )
