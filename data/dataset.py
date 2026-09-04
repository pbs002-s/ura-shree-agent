"""
PyTorch Dataset and DataLoader implementations for Causal Language Modeling in URA-Shree.
Supports memory-mapped binary files (np.memmap) for streaming multi-gigabyte datasets
with zero RAM overhead and instant random batch access.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union, Tuple


class CausalLMDataset(Dataset):
    """
    Autoregressive Next-Token Prediction Dataset.
    Given a sequence of token IDs, slices pairs of (x, y) where:
        x = tokens[i : i + seq_len]
        y = tokens[i + 1 : i + seq_len + 1]
    y[t] is the ground-truth next token for x[t].
    """

    def __init__(
        self,
        data_source: Union[str, np.ndarray, torch.Tensor],
        seq_len: int = 1024,
        dtype: np.dtype = np.uint16,
    ):
        """
        Args:
            data_source: Filepath to a .bin file, or an in-memory numpy array / torch tensor.
            seq_len: Context window size (T).
            dtype: Data type stored in the binary file (default np.uint16 supports vocabs <= 65535).
        """
        self.seq_len = seq_len

        if isinstance(data_source, str):
            if not os.path.exists(data_source):
                raise FileNotFoundError(f"Dataset file does not exist: {data_source}")
            # Memory-map the binary file for zero-copy high-throughput disk access
            self.data = np.memmap(data_source, dtype=dtype, mode="r")
            self.total_tokens = len(self.data)
        elif isinstance(data_source, np.ndarray):
            self.data = data_source
            self.total_tokens = len(self.data)
        elif isinstance(data_source, torch.Tensor):
            self.data = data_source.cpu().numpy().astype(dtype)
            self.total_tokens = len(self.data)
        else:
            raise TypeError(f"Unsupported data source type: {type(data_source)}")

        # Total number of non-overlapping sequence samples available
        # Need at least seq_len + 1 tokens to construct one (x, y) pair
        if self.total_tokens <= self.seq_len:
            raise ValueError(
                f"Total tokens ({self.total_tokens}) must be greater than seq_len ({self.seq_len}) to produce samples."
            )

        self.num_samples = (self.total_tokens - 1) // self.seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches the (x, y) sample at the given index.
        Returns:
            x: torch.LongTensor of shape [seq_len]
            y: torch.LongTensor of shape [seq_len]
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset with {self.num_samples} samples.")

        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len

        # Slice data from memory map or array
        chunk = self.data[start_idx : end_idx + 1].astype(np.int64)

        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])

        return x, y

    def close(self) -> None:
        """Explicitly closes underlying memory map file handles (crucial for Windows)."""
        if hasattr(self, "data") and isinstance(self.data, np.memmap):
            if hasattr(self.data, "_mmap") and self.data._mmap is not None:
                self.data._mmap.close()
            del self.data


def create_dataloader(
    dataset: CausalLMDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Creates a high-efficiency PyTorch DataLoader for training or evaluation.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
    )
