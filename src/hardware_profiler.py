
import platform
import os
import shutil
import subprocess

import psutil


def get_hardware_summary():
    """Detect system resources and available AI acceleration."""

    memory = psutil.virtual_memory()

    info = {
        "operating_system": platform.platform(),
        "processor": platform.processor() or "Unknown processor",
        "cpu_cores": psutil.cpu_count(logical=False) or 1,
        "logical_cores": psutil.cpu_count(logical=True) or 1,
        "total_ram_gb": round(memory.total / (1024 ** 3), 2),
        "available_ram_gb": round(memory.available / (1024 ** 3), 2),
        "gpu_available": False,
        "gpu_name": "Not detected",
        "gpu_memory_gb": None,
        "gpu_vendor": "Unknown",
        "cuda_available": False,
        "mps_available": False,
        "recommended_runtime": "CPU",
    }

    # Detect NVIDIA GPU and CUDA support through PyTorch.
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["cuda_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_vendor"] = "NVIDIA"

            properties = torch.cuda.get_device_properties(0)
            info["gpu_memory_gb"] = round(
                properties.total_memory / (1024 ** 3), 2
            )
            info["recommended_runtime"] = "CUDA"

        # Detect Apple's Metal Performance Shaders backend.
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            info["gpu_available"] = True
            info["mps_available"] = True
            info["gpu_name"] = "Apple GPU (Metal)"
            info["gpu_vendor"] = "Apple"
            info["recommended_runtime"] = "MPS"

    except (ImportError, RuntimeError, OSError):
        pass

    # Additional NVIDIA detection when PyTorch cannot detect CUDA.
    if not info["gpu_available"] and shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )

            first_gpu = result.stdout.strip().splitlines()[0]
            parts = first_gpu.split(",")

            if len(parts) >= 2:
                info["gpu_name"] = parts[0].strip()
                info["gpu_vendor"] = "NVIDIA"
                info["gpu_memory_gb"] = round(
                    float(parts[1].strip()) / 1024, 2
                )
                info["gpu_available"] = True
                info["recommended_runtime"] = (
                    "CUDA (driver detected; PyTorch support unconfirmed)"
                )

        except (
            subprocess.SubprocessError,
            OSError,
            ValueError,
            IndexError,
        ):
            pass

    # Conservative baseline recommendation based on available RAM.
    # This does not guarantee that a model will fit.
    available_ram = info["available_ram_gb"]

    if info["recommended_runtime"] == "CUDA":
        info["inference_advice"] = (
            "Consider a quantized model sized for available VRAM; "
            "benchmark it before selecting a configuration."
        )
    elif info["recommended_runtime"] == "MPS":
        info["inference_advice"] = (
            "Consider a quantized model supported by the Apple runtime "
            "and benchmark it locally."
        )
    elif available_ram < 2:
        info["inference_advice"] = (
            "Very limited available RAM. Avoid loading an LLM until "
            "memory availability improves."
        )
    elif available_ram < 4:
        info["inference_advice"] = (
            "Try a very small quantized model, but monitor memory closely."
        )
    elif available_ram < 8:
        info["inference_advice"] = (
            "A small quantized model may work on CPU; test memory and speed."
        )
    else:
        info["inference_advice"] = (
            "CPU inference may be feasible with a suitable quantized model; "
            "benchmark before making a recommendation."
        )

    return info
