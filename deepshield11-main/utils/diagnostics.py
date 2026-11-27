# Purpose: Provides deep system diagnostics, environmental checks, hardware stats, and startup verification.

import os
import sys
import shutil
import subprocess
import platform
import time
from typing import Dict, Any

import torch

def ensure_directories():
    """Ensures all required project directories exist."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    required_dirs = [
        os.path.join(base_dir, "reports"),
        os.path.join(base_dir, "uploads"),
        os.path.join(base_dir, "logs"),
        os.path.join(base_dir, "temp")
    ]
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)
    return required_dirs

def check_ffmpeg() -> Dict[str, Any]:
    """Checks if FFmpeg binary is installed and accessible in PATH."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        possible_paths = [
            "C:\\ffmpeg\\bin\\ffmpeg.exe",
            "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
            "C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                ffmpeg_path = p
                break

    if ffmpeg_path:
        try:
            res = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True, timeout=3)
            version_line = res.stdout.splitlines()[0] if res.stdout else "FFmpeg installed"
            return {"available": True, "path": ffmpeg_path, "version": version_line}
        except Exception as e:
            return {"available": False, "path": ffmpeg_path, "error": str(e)}
    else:
        return {"available": False, "path": None, "warning": "FFmpeg not found in PATH. Native audio extraction fallbacks will be used."}

def check_torch_hardware() -> Dict[str, Any]:
    """Inspects PyTorch runtime and CUDA/CPU hardware state."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Host Processor)"
    device_count = torch.cuda.device_count() if cuda_available else 0

    return {
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "device_count": device_count,
        "threads": torch.get_num_threads()
    }

def get_system_diagnostics() -> Dict[str, Any]:
    """Returns a complete diagnostic report of the server environment."""
    ensure_directories()
    torch_info = check_torch_hardware()
    ffmpeg_info = check_ffmpeg()

    try:
        import psutil
        mem = psutil.virtual_memory()
        mem_info = {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "used_percent": mem.percent
        }
    except Exception:
        mem_info = {"status": "available"}

    disk = shutil.disk_usage(os.path.abspath(os.path.dirname(__file__)))

    return {
        "status": "online",
        "service": "DeepShield AI Engine",
        "version": "3.5.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "hardware": torch_info,
        "ffmpeg": ffmpeg_info,
        "memory": mem_info,
        "disk": {
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "used_percent": round((disk.used / disk.total) * 100, 1)
        }
    }

if __name__ == "__main__":
    import json
    print(json.dumps(get_system_diagnostics(), indent=2))
