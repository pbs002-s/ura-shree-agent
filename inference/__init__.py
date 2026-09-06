"""Local inference engine and runtime tuning."""
try:
    from inference.engine import InferenceEngine, GenerationStats
    from inference.runtime import RuntimeProfile, tune_runtime, gpu_memory_snapshot, host_memory_snapshot
except ImportError:
    # Allows outer launcher scripts or virtualenv delegates to initialize without crashing
    InferenceEngine = None  # type: ignore
    GenerationStats = None  # type: ignore
    RuntimeProfile = None  # type: ignore
    tune_runtime = None  # type: ignore
    gpu_memory_snapshot = None  # type: ignore
    host_memory_snapshot = None  # type: ignore

__all__ = [
    "InferenceEngine",
    "GenerationStats",
    "RuntimeProfile",
    "tune_runtime",
    "gpu_memory_snapshot",
    "host_memory_snapshot",
]

