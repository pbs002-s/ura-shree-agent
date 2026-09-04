"""
Unit tests for environment readiness and configuration integrity.
"""

import os
import sys
import pytest

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.config import ModelConfig, TrainingConfig, CheckpointConfig, ProjectConfig


def test_pytorch_installation():
    """Verify PyTorch is installed and accessible."""
    import torch
    assert torch.__version__ is not None
    # Verify basic tensor creation
    t = torch.tensor([1.0, 2.0, 3.0])
    assert t.shape == (3,)


def test_cuda_capability():
    """Verify CUDA availability if an NVIDIA GPU is present."""
    import torch
    if torch.cuda.is_available():
        assert torch.cuda.device_count() > 0
        device_name = torch.cuda.get_device_name(0)
        assert len(device_name) > 0
        # Test basic CUDA tensor operation
        x = torch.ones(10, 10, device="cuda")
        y = x * 2.0
        assert torch.all(y == 2.0).item()


def test_model_config_validation():
    """Test model configuration rules and constraints."""
    cfg = ModelConfig(
        vocab_size=8192,
        max_seq_len=1024,
        embed_dim=384,
        num_layers=6,
        num_heads=6,
        intermediate_dim=1536
    )
    assert cfg.head_dim == 64
    assert cfg.to_dict()["vocab_size"] == 8192

    # Assert embed_dim divisible by num_heads
    with pytest.raises(AssertionError):
        ModelConfig(embed_dim=384, num_heads=7)


def test_load_small_yaml():
    """Verify configs/small.yaml loads and produces a valid ProjectConfig."""
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "configs", "small.yaml")
    assert os.path.exists(yaml_path), f"Configuration file not found at {yaml_path}"

    config = ProjectConfig.load_from_yaml(yaml_path)
    # The vocabulary matches the tokenizer trained on the source tree.
    assert config.model.vocab_size == 4096
    assert config.model.num_layers == 6
    assert config.model.num_heads == 6
    assert config.training.batch_size == 16
    assert config.checkpoint.dir == "checkpoints"

    # New runs use rotary positions, SwiGLU and grouped-query attention.
    assert config.model.pos_encoding == "rope"
    assert config.model.ffn == "swiglu"
    assert config.model.effective_kv_heads == 3


def test_load_medium_and_large_yamls():
    """Verify configs/medium.yaml and configs/large.yaml load cleanly."""
    configs_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
    
    med_cfg = ProjectConfig.load_from_yaml(os.path.join(configs_dir, "medium.yaml"))
    assert med_cfg.model.embed_dim == 768
    assert med_cfg.model.num_layers == 12
    assert med_cfg.model.head_dim == 64

    large_cfg = ProjectConfig.load_from_yaml(os.path.join(configs_dir, "large.yaml"))
    assert large_cfg.model.embed_dim == 1024
    assert large_cfg.model.num_layers == 24
    assert large_cfg.model.head_dim == 64

