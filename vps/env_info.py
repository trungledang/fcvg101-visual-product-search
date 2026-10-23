"""Record runtime hardware/software facts (observed, not assumed)."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e.__class__.__name__}"


def collect(path_for_disk: str = ".") -> dict:
    import numpy, PIL, torch, torchvision  # noqa: E401
    mem_total = None
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemTotal"):
                    mem_total = int(ln.split()[1]) * 1024
    except OSError:
        pass
    du = shutil.disk_usage(path_for_disk)
    cpu_model = ""
    try:
        with open("/proc/cpuinfo") as f:
            cpu_model = next((ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")), "")
    except OSError:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(mem_total / 2**30, 2) if mem_total else None,
        "disk_free_gb": round(du.free / 2**30, 2),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "nvidia_smi": _cmd(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
    }


def peak_rss_mb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
