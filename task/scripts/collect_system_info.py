#!/usr/bin/env python3
"""collect_system_info.py

Gathers complete hardware, GPU, CUDA, OS, and Python package environment diagnostics.
Used for reproducibility, debugging, and system verification reports.
"""

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def get_gpu_info() -> Dict[str, Any]:
    """Retrieve GPU information using PyTorch or nvidia-smi."""
    info: Dict[str, Any] = {"cuda_available": False, "devices": []}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"
        info["device_count"] = torch.cuda.device_count()

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                cc = torch.cuda.get_device_capability(i)
                info["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "multi_processor_count": props.multi_processor_count,
                    "compute_capability": f"{cc[0]}.{cc[1]}",
                    "turing_t4_optimized": cc == (7, 5),
                })
    except Exception as e:
        info["error"] = str(e)

    # Check nvidia-smi command
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=gpu_name,driver_version,memory.total,memory.free,memory.used", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if res.returncode == 0:
                info["nvidia_smi_output"] = res.stdout.strip()
        except Exception:
            pass

    return info


def get_package_versions() -> Dict[str, str]:
    """Inspect versions of installed deep learning packages."""
    packages = [
        "torch",
        "transformers",
        "peft",
        "trl",
        "bitsandbytes",
        "accelerate",
        "datasets",
        "tokenizers",
        "yaml",
        "pydantic",
        "rich",
        "numpy",
    ]
    installed = {}
    for pkg in packages:
        imp_name = "yaml" if pkg == "yaml" else pkg
        try:
            mod = importlib.import_module(imp_name)
            installed[pkg] = getattr(mod, "__version__", "Installed (no __version__)")
        except ImportError:
            installed[pkg] = "Not installed"
    return installed


def generate_system_summary() -> str:
    """Generate a clean ASCII formatted system diagnostic report."""
    gpu = get_gpu_info()
    pkgs = get_package_versions()

    lines = [
        "=" * 78,
        " SYSTEM & HARDWARE DIAGNOSTICS REPORT",
        "=" * 78,
        f"OS / Platform:          {platform.system()} {platform.release()} ({platform.machine()})",
        f"Python Version:         {platform.python_version()} ({sys.executable})",
        f"PyTorch CUDA Built:     {gpu.get('torch_cuda_version', 'N/A')}",
        f"cuDNN Version:          {gpu.get('cudnn_version', 'N/A')}",
        f"CUDA Available:         {gpu.get('cuda_available', False)}",
        "-" * 78,
        "ACCELERATOR / GPU DEVICES:",
    ]

    if gpu.get("devices"):
        for dev in gpu["devices"]:
            t4_badge = "[TURING T4 DETECTED - FP16 / QLORA READY]" if dev.get("turing_t4_optimized") else ""
            lines.append(f"  * GPU {dev['index']}: {dev['name']} ({dev['total_memory_gb']} GB VRAM) | CC {dev['compute_capability']} {t4_badge}")
    else:
        lines.append("  * No active CUDA GPU detected via PyTorch. (Fallback to CPU or Cloud runner)")

    if "nvidia_smi_output" in gpu and gpu["nvidia_smi_output"]:
        lines.append(f"  * nvidia-smi query: {gpu['nvidia_smi_output']}")

    lines.extend([
        "-" * 78,
        "DEEP LEARNING SOFTWARE STACK:",
    ])
    for p, ver in pkgs.items():
        lines.append(f"  * {p:<18}: {ver}")

    lines.extend([
        "=" * 78,
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Collect system and GPU diagnostics.")
    parser.add_argument("--output-log", type=str, default=None, help="Save text report to file")
    parser.add_argument("--output-json", type=str, default=None, help="Save JSON report to file")
    args = parser.parse_args()

    report = generate_system_summary()
    print(report)

    if args.output_log:
        p = Path(args.output_log)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report, encoding="utf-8")
        print(f"Diagnostics written to: {p}")

    if args.output_json:
        pj = Path(args.output_json)
        pj.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu": get_gpu_info(),
            "packages": get_package_versions(),
        }
        pj.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"JSON diagnostics written to: {pj}")


if __name__ == "__main__":
    main()
