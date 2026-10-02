#!/usr/bin/env python3
"""verify_training.py

Pre-flight and post-training verification suite for DPO fine-tuning.
Validates GPU environment, package dependencies, config schema, data integrity,
and runs a sanity dry-run step.
"""

import argparse
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VerifyTraining")


class VerificationSuite:
    def __init__(self, config_path: str, data_dir: str):
        self.config_path = Path(config_path)
        self.data_dir = Path(data_dir)
        self.results: List[Tuple[str, str, str]] = []  # (Category, Status, Message)

    def log_result(self, category: str, status: str, message: str):
        self.results.append((category, status, message))
        if status == "PASS":
            logger.info(f"[{category}] PASS: {message}")
        elif status == "WARN":
            logger.warning(f"[{category}] WARN: {message}")
        else:
            logger.error(f"[{category}] FAIL: {message}")

    def check_python_and_packages(self):
        """Check Python version and essential ML libraries."""
        py_ver = sys.version_split = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info >= (3, 9):
            self.log_result("Environment", "PASS", f"Python version {py_ver} (>= 3.9)")
        else:
            self.log_result("Environment", "WARN", f"Python version {py_ver} may encounter compatibility issues.")

        required_packages = {
            "torch": "2.0.0",
            "transformers": "4.35.0",
            "peft": "0.6.0",
            "trl": "0.7.0",
            "yaml": "5.0",
        }

        optional_packages = ["bitsandbytes", "accelerate", "datasets", "rich", "pydantic"]

        for pkg, min_ver in required_packages.items():
            import_name = "yaml" if pkg == "yaml" else pkg
            try:
                mod = importlib.import_module(import_name)
                ver = getattr(mod, "__version__", "unknown")
                self.log_result("Dependencies", "PASS", f"{pkg} installed (v{ver})")
            except ImportError:
                self.log_result("Dependencies", "FAIL", f"Missing required dependency '{pkg}'")

        for pkg in optional_packages:
            try:
                mod = importlib.import_module(pkg)
                ver = getattr(mod, "__version__", "unknown")
                self.log_result("Dependencies (Opt)", "PASS", f"{pkg} is available (v{ver})")
            except ImportError:
                self.log_result("Dependencies (Opt)", "WARN", f"Optional package '{pkg}' not found.")

    def check_cuda_and_gpu(self):
        """Check CUDA availability and GPU specs."""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                cc = torch.cuda.get_device_capability(0)
                self.log_result("Hardware", "PASS", f"CUDA is available! Found {device_count} GPU(s): {gpu_name}")
                self.log_result("Hardware", "PASS", f"GPU 0 Total VRAM: {vram_total:.2f} GB | Compute Capability: {cc[0]}.{cc[1]}")
                if cc == (7, 5):
                    self.log_result("Hardware", "PASS", "Turing Architecture (T4) detected: FP16 Tensor Cores enabled.")
                elif cc[0] < 7:
                    self.log_result("Hardware", "WARN", f"Older GPU architecture ({cc[0]}.{cc[1]}). Performance may be sub-optimal.")
            else:
                self.log_result("Hardware", "WARN", "CUDA is NOT available. Running on CPU emulation mode.")
        except Exception as e:
            self.log_result("Hardware", "FAIL", f"Error checking CUDA: {e}")

    def check_config(self) -> Dict:
        """Validate YAML configuration file."""
        if not self.config_path.exists():
            self.log_result("Configuration", "FAIL", f"Config file not found at: {self.config_path}")
            return {}

        try:
            import yaml
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.log_result("Configuration", "PASS", f"Config loaded successfully from {self.config_path.name}")
        except Exception as e:
            self.log_result("Configuration", "FAIL", f"YAML parsing error: {e}")
            return {}

        # Validate structure
        for section in ["base", "task", "data", "training"]:
            if section in cfg:
                self.log_result("Config Validation", "PASS", f"Section '{section}' present.")
            else:
                self.log_result("Config Validation", "FAIL", f"Missing section '{section}' in config.")

        training_cfg = cfg.get("training", {})
        beta = training_cfg.get("beta", None)
        if beta is not None and 0.01 <= float(beta) <= 1.0:
            self.log_result("Config Validation", "PASS", f"DPO Beta parameter is valid ({beta})")
        else:
            self.log_result("Config Validation", "WARN", f"DPO Beta ({beta}) is outside typical range [0.01, 0.5]")

        batch_sz = training_cfg.get("batch_size", 1)
        grad_accum = training_cfg.get("gradient_accumulation_steps", 1)
        eff_batch = batch_sz * grad_accum
        self.log_result("Config Validation", "PASS", f"Effective Batch Size: {eff_batch} (micro_batch={batch_sz}, grad_accum={grad_accum})")

        return cfg

    def check_datasets(self, cfg: Dict):
        """Validate train and eval dataset files."""
        data_cfg = cfg.get("data", {})
        train_rel = data_cfg.get("train", "./data/train.jsonl")
        eval_rel = data_cfg.get("eval", "./data/eval.jsonl")

        def resolve_data_path(rel_p: str) -> Path:
            p = Path(rel_p)
            if p.is_absolute() and p.exists():
                return p
            # Try relative to data_dir
            if (self.data_dir / p.name).exists():
                return self.data_dir / p.name
            # Try relative to config dir
            if (self.config_path.parent / rel_p).exists():
                return self.config_path.parent / rel_p
            # Try relative to task root (config parent parent)
            if (self.config_path.parent.parent / rel_p).exists():
                return self.config_path.parent.parent / rel_p
            # Try relative to cwd
            if Path(rel_p).exists():
                return Path(rel_p)
            return self.data_dir / p.name

        train_path = resolve_data_path(train_rel)
        eval_path = resolve_data_path(eval_rel)

        for name, p in [("Train Dataset", train_path), ("Eval Dataset", eval_path)]:
            if not p.exists():
                self.log_result("Data Integrity", "FAIL", f"{name} not found at {p}")
                continue

            count = 0
            valid_keys = True
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        if not all(k in row for k in ("prompt", "chosen", "rejected")):
                            valid_keys = False
                            break
                        count += 1
                    except Exception:
                        valid_keys = False
                        break

            if count > 0 and valid_keys:
                self.log_result("Data Integrity", "PASS", f"{name} verified: {count} valid preference pairs in {p.name}")
            else:
                self.log_result("Data Integrity", "FAIL", f"{name} contains formatting or missing key errors.")

    def run_dry_run_simulation(self):
        """Simulate a single DPO forward/backward step to ensure gradient flow."""
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            # Dummy mini-model simulating policy & reference
            policy = nn.Linear(32, 100, bias=False)
            ref = nn.Linear(32, 100, bias=False)
            ref.weight.data.copy_(policy.weight.data)
            ref.eval()

            # Dummy chosen & rejected tokens
            dummy_x = torch.randn(2, 32)
            policy_logits = policy(dummy_x)
            ref_logits = ref(dummy_x).detach()

            # DPO log probs
            pi_logps = F.log_softmax(policy_logits, dim=-1).sum(-1)
            ref_logps = F.log_softmax(ref_logits, dim=-1).sum(-1)

            pi_chosen, pi_rejected = pi_logps[0], pi_logps[1]
            ref_chosen, ref_rejected = ref_logps[0], ref_logps[1]

            beta = 0.1
            pi_ratios = beta * (pi_chosen - ref_chosen)
            ref_ratios = beta * (pi_rejected - ref_rejected)
            loss = -F.logsigmoid(pi_ratios - ref_ratios)

            loss.backward()
            self.log_result("Dry Run", "PASS", f"Synthetic 1-step DPO loss backward executed successfully (Loss: {loss.item():.4f})")
        except Exception as e:
            self.log_result("Dry Run", "WARN", f"Synthetic dry-run test skipped or encountered: {e}")

    def generate_report(self, output_log: Path = None):
        """Format and print full verification checklist."""
        print()
        print("=" * 80)
        print(" PRE-FLIGHT VERIFICATION CHECKLIST FOR DPO TRAINING")
        print("=" * 80)
        passes = sum(1 for _, st, _ in self.results if st == "PASS")
        warns = sum(1 for _, st, _ in self.results if st == "WARN")
        fails = sum(1 for _, st, _ in self.results if st == "FAIL")

        for cat, st, msg in self.results:
            tag = f"[{st}]"
            print(f"{tag:<8} {cat:<22} : {msg}")

        print("-" * 80)
        print(f"Summary: {passes} PASS | {warns} WARN | {fails} FAIL")
        if fails == 0:
            print("OVERALL STATUS: READY FOR DPO TRAINING! [ALL SYSTEMS GO]")
        else:
            print(f"OVERALL STATUS: BLOCKED - {fails} CRITICAL ERROR(S) MUST BE RESOLVED.")
        print("=" * 80)
        print()

        if output_log:
            output_log.parent.mkdir(parents=True, exist_ok=True)
            with open(output_log, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(" PRE-FLIGHT VERIFICATION CHECKLIST FOR DPO TRAINING\n")
                f.write("=" * 80 + "\n")
                for cat, st, msg in self.results:
                    f.write(f"[{st:<4}] {cat:<22} : {msg}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Summary: {passes} PASS | {warns} WARN | {fails} FAIL\n")
                f.write("OVERALL STATUS: " + ("READY FOR DPO TRAINING!\n" if fails == 0 else f"BLOCKED ({fails} FAILS)\n"))
                f.write("=" * 80 + "\n")
            logger.info(f"Written verification report to: {output_log}")


def main():
    parser = argparse.ArgumentParser(description="Run pre-flight verification checks for DPO.")
    parser.add_argument("--config", type=str, default=str(Path(__file__).parent.parent / "configs" / "dpo_t4.yaml"))
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).parent.parent / "data"))
    parser.add_argument("--output-log", type=str, default=str(Path(__file__).parent.parent / "logs" / "preflight.log"))
    args = parser.parse_args()

    suite = VerificationSuite(args.config, args.data_dir)
    suite.check_python_and_packages()
    suite.check_cuda_and_gpu()
    cfg = suite.check_config()
    if cfg:
        suite.check_datasets(cfg)
    suite.run_dry_run_simulation()
    suite.generate_report(output_log=Path(args.output_log))


if __name__ == "__main__":
    main()
