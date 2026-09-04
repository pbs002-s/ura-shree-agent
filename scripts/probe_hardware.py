"""
Hardware & Environment Diagnostic Script for URA-Shree.
Verifies Python, CUDA, PyTorch, GPU VRAM, and tensor compute capabilities.
"""

import sys
import platform
import psutil

def probe_environment():
    print("=" * 65)
    print("        URA-Shree: Hardware & System Diagnostic Probe")
    print("=" * 65)

    # 1. Host System Info
    print(f"[OS] Platform          : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"[OS] Python Version    : {sys.version.split()[0]}")
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    ram_avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    print(f"[RAM] System Total/Avail: {ram_gb:.2f} GB / {ram_avail_gb:.2f} GB")
    print(f"[CPU] Physical / Logical: {psutil.cpu_count(logical=False)} / {psutil.cpu_count(logical=True)}")

    # 2. PyTorch & CUDA Detection
    try:
        import torch
        print(f"\n[PyTorch] Version       : {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"[CUDA] Available       : {cuda_available}")

        if cuda_available:
            device_count = torch.cuda.device_count()
            device_idx = 0
            device_name = torch.cuda.get_device_name(device_idx)
            cuda_version = torch.version.cuda
            vram_gb = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
            capability = torch.cuda.get_device_capability(device_idx)
            bf16_supported = torch.cuda.is_bf16_supported()

            print(f"[CUDA] Version         : {cuda_version}")
            print(f"[GPU] Device [{device_idx}]       : {device_name}")
            print(f"[GPU] Total VRAM       : {vram_gb:.2f} GB")
            print(f"[GPU] Compute Cap.     : {capability[0]}.{capability[1]}")
            print(f"[GPU] BF16 Native Accel: {bf16_supported}")

            # 3. Perform sanity tensor operation on GPU
            print("\n[Self-Test] Running GPU matrix multiplication test...")
            a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
            b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            print(f"[Self-Test] Success! Tensor shape: {list(c.shape)}, Device: {c.device}, dtype: {c.dtype}")
        else:
            print("\n[WARNING] CUDA is not available. PyTorch will run in CPU mode.")
            print("[Self-Test] Running CPU matrix multiplication test...")
            a = torch.randn(512, 512, dtype=torch.float32)
            b = torch.randn(512, 512, dtype=torch.float32)
            c = torch.matmul(a, b)
            print(f"[Self-Test] Success! Tensor shape: {list(c.shape)}, Device: {c.device}")

    except ImportError as e:
        print(f"\n[ERROR] PyTorch is not yet installed in this environment: {e}")
        return False

    print("=" * 65)
    print("Diagnostic Complete: System is ready for URA-Shree training & inference.")
    print("=" * 65)
    return True

if __name__ == "__main__":
    success = probe_environment()
    sys.exit(0 if success else 1)
