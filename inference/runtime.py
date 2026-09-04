"""
Hardware detection and runtime tuning for local inference.

The goal is a single call - `tune_runtime()` - that inspects the machine and
sets the handful of PyTorch knobs that actually move the needle, so the local
model runs near the hardware's practical ceiling without the caller thinking
about it.

What gets decided here:

  * Device selection (CUDA / Apple MPS / CPU).
  * Compute dtype. bfloat16 on Ampere and newer, float16 on older CUDA cards,
    float32 on CPU (CPU float16 is emulated and slower, not faster).
  * TF32 matmul on CUDA, which is a free ~2x on fp32 GEMMs with no accuracy
    cost that matters for inference.
  * CPU thread count. PyTorch defaults to one thread per logical core, which on
    a hyperthreaded laptop oversubscribes and thrashes. Physical cores wins.
  * Dynamic int8 quantisation of the linear layers on CPU, which cuts resident
    weight memory roughly 4x against float32 and speeds up the GEMMs.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import torch

try:
    import psutil
except ImportError:  # psutil is listed in requirements but stay soft on it
    psutil = None  # type: ignore


@dataclass
class RuntimeProfile:
    """The resolved execution environment for a loaded model."""

    device: str
    device_name: str
    dtype: str
    cpu_threads: int
    physical_cores: int
    logical_cores: int
    total_ram_gb: float
    available_ram_gb: float
    total_vram_gb: float
    compute_capability: Optional[str]
    tf32_enabled: bool
    quantized: bool
    notes: list

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _core_counts() -> tuple:
    logical = os.cpu_count() or 1
    physical = logical
    if psutil is not None:
        try:
            physical = psutil.cpu_count(logical=False) or logical
        except Exception:
            physical = logical
    return physical, logical


def _ram_gb() -> tuple:
    if psutil is None:
        return 0.0, 0.0
    try:
        vm = psutil.virtual_memory()
        return round(vm.total / 1e9, 2), round(vm.available / 1e9, 2)
    except Exception:
        return 0.0, 0.0


def pick_device(preference: Optional[str] = None) -> torch.device:
    """Resolves a device string, falling back gracefully when the ask is unavailable."""
    if preference:
        pref = preference.lower()
        if pref.startswith("cuda") and torch.cuda.is_available():
            return torch.device(pref)
        if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        if pref == "cpu":
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    """
    Widest-but-fastest dtype the device handles well.

    bfloat16 needs compute capability 8.0+ (Ampere). Below that, float16 is the
    fast path. On CPU, float32 stays: half precision there is emulated.
    """
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def tune_runtime(
    device_preference: Optional[str] = None,
    thread_budget: Optional[int] = None,
) -> tuple:
    """
    Applies global PyTorch settings and returns (device, dtype, RuntimeProfile).

    Safe to call more than once; every setting written here is idempotent.
    """
    notes: list = []
    device = pick_device(device_preference)
    dtype = pick_dtype(device)

    physical, logical = _core_counts()
    total_ram, avail_ram = _ram_gb()
    total_vram = 0.0
    capability = None
    tf32 = False

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        device_name = props.name
        total_vram = round(props.total_memory / 1e9, 2)
        capability = f"{props.major}.{props.minor}"

        # TF32 turns fp32 matmuls into tensor-core ops. Free speed on Ampere+.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        tf32 = True
        notes.append("TF32 matmul and cuDNN autotuning enabled")

        if hasattr(torch.backends.cuda, "matmul") and hasattr(
            torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"
        ):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    else:
        device_name = platform.processor() or platform.machine() or "CPU"

    # One thread per physical core. Oversubscribing logical cores makes small
    # GEMMs slower, not faster, because the threads fight over shared L2.
    threads = thread_budget or max(1, physical)
    threads = min(threads, logical)
    try:
        torch.set_num_threads(threads)
        # Inter-op parallelism can only be set before any parallel work starts.
        torch.set_num_interop_threads(max(1, min(2, threads)))
    except RuntimeError:
        # Already initialised by an earlier call; the existing value stands.
        threads = torch.get_num_threads()
    notes.append(f"CPU threads pinned to {threads} of {logical} logical / {physical} physical")

    if device.type == "cpu":
        # Keep allocator arenas from ballooning on long-running servers.
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(threads))

    profile = RuntimeProfile(
        device=str(device),
        device_name=device_name,
        dtype=str(dtype).replace("torch.", ""),
        cpu_threads=threads,
        physical_cores=physical,
        logical_cores=logical,
        total_ram_gb=total_ram,
        available_ram_gb=avail_ram,
        total_vram_gb=total_vram,
        compute_capability=capability,
        tf32_enabled=tf32,
        quantized=False,
        notes=notes,
    )
    return device, dtype, profile


def quantize_for_cpu(model: torch.nn.Module) -> tuple:
    """
    Dynamic int8 quantisation of every nn.Linear.

    Weights are stored as int8 and activations are quantised on the fly, so
    resident memory drops about 4x against float32 while accuracy loss on a
    decoder LM is small. CPU only - CUDA has no dynamic-quant kernels.

    Returns (model, applied) so the caller can report honestly when it is a no-op.
    """
    try:
        quantized = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return quantized, True
    except Exception:
        return model, False


def gpu_memory_snapshot() -> Dict[str, float]:
    """Live VRAM figures in MB, or zeros when there is no CUDA device."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0, "free_mb": 0.0}
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1),
        "total_mb": round(total_b / 1e6, 1),
        "free_mb": round(free_b / 1e6, 1),
    }


def host_memory_snapshot() -> Dict[str, float]:
    """Resident set size of this process plus system-wide RAM, in MB."""
    if psutil is None:
        return {"process_rss_mb": 0.0, "system_used_mb": 0.0, "system_total_mb": 0.0}
    try:
        proc = psutil.Process()
        vm = psutil.virtual_memory()
        return {
            "process_rss_mb": round(proc.memory_info().rss / 1e6, 1),
            "system_used_mb": round((vm.total - vm.available) / 1e6, 1),
            "system_total_mb": round(vm.total / 1e6, 1),
        }
    except Exception:
        return {"process_rss_mb": 0.0, "system_used_mb": 0.0, "system_total_mb": 0.0}
