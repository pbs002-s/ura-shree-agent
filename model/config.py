"""
Configuration classes for URA-Shree Language Model & Agent.
Handles loading, validation, and serialization of model and training parameters.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import yaml
import os


@dataclass
class ModelConfig:
    """Hyperparameters defining the URA-Shree decoder-only Transformer architecture."""
    name: str = "ura-shree-small"
    vocab_size: int = 8192
    max_seq_len: int = 1024
    embed_dim: int = 384
    num_layers: int = 6
    num_heads: int = 6
    intermediate_dim: int = 1536
    dropout: float = 0.1
    tie_weights: bool = True
    norm_eps: float = 1e-5
    bias: bool = False

    # --- Architecture switches -------------------------------------------------
    # Defaults reproduce the original architecture exactly, so checkpoints saved
    # before these fields existed still load and evaluate identically.
    pos_encoding: str = "learned"      # "learned" | "rope"
    rope_theta: float = 10000.0
    num_kv_heads: int = 0              # 0 or == num_heads means plain MHA
    ffn: str = "gelu"                  # "gelu" | "swiglu"

    def __post_init__(self):
        self.vocab_size = int(self.vocab_size)
        self.max_seq_len = int(self.max_seq_len)
        self.embed_dim = int(self.embed_dim)
        self.num_layers = int(self.num_layers)
        self.num_heads = int(self.num_heads)
        self.dropout = float(self.dropout)
        self.norm_eps = float(self.norm_eps)
        assert self.embed_dim % self.num_heads == 0, (
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
        )
        if self.intermediate_dim is None or int(self.intermediate_dim) <= 0:
            self.intermediate_dim = 4 * self.embed_dim
        else:
            self.intermediate_dim = int(self.intermediate_dim)

        self.pos_encoding = str(self.pos_encoding).lower()
        assert self.pos_encoding in ("learned", "rope"), (
            f"pos_encoding must be 'learned' or 'rope', got {self.pos_encoding!r}"
        )
        self.ffn = str(self.ffn).lower()
        assert self.ffn in ("gelu", "swiglu"), f"ffn must be 'gelu' or 'swiglu', got {self.ffn!r}"

        self.num_kv_heads = int(self.num_kv_heads or 0)
        if self.num_kv_heads:
            assert self.num_heads % self.num_kv_heads == 0, (
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            )
        if self.pos_encoding == "rope":
            assert self.head_dim % 2 == 0, "RoPE requires an even head_dim"

    @property
    def effective_kv_heads(self) -> int:
        """Key/value head count actually used by attention (num_heads when GQA is off)."""
        return self.num_kv_heads or self.num_heads

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


@dataclass
class TrainingConfig:
    """Hyperparameters for training optimization, scheduling, and device management."""
    batch_size: int = 16
    grad_accum_steps: int = 4
    learning_rate: float = 6.0e-4
    min_learning_rate: float = 6.0e-5
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50
    save_interval: int = 500
    grad_clip: float = 1.0
    mixed_precision: bool = True
    device: str = "cuda"
    seed: int = 42

    def __post_init__(self):
        self.batch_size = int(self.batch_size)
        self.grad_accum_steps = int(self.grad_accum_steps)
        self.learning_rate = float(self.learning_rate)
        self.min_learning_rate = float(self.min_learning_rate)
        self.weight_decay = float(self.weight_decay)
        self.warmup_steps = int(self.warmup_steps)
        self.max_steps = int(self.max_steps)
        self.eval_interval = int(self.eval_interval)
        self.eval_iters = int(self.eval_iters)
        self.save_interval = int(self.save_interval)
        self.grad_clip = float(self.grad_clip)
        self.seed = int(self.seed)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


@dataclass
class CheckpointConfig:
    """Configuration for model weight saving and restoration."""
    dir: str = "checkpoints"
    save_best: bool = True
    keep_last: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


@dataclass
class ProjectConfig:
    """Unified configuration encapsulating model, training, and checkpoint parameters."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    @classmethod
    def load_from_yaml(cls, path: str) -> "ProjectConfig":
        """Load and validate configuration from a YAML file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        model_cfg = ModelConfig.from_dict(raw.get("model", {}))
        training_cfg = TrainingConfig.from_dict(raw.get("training", {}))
        checkpoint_cfg = CheckpointConfig.from_dict(raw.get("checkpoint", {}))

        return cls(model=model_cfg, training=training_cfg, checkpoint=checkpoint_cfg)

    def save_to_yaml(self, path: str) -> None:
        """Save configuration state to a YAML file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
