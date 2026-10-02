# 🍲 Soup AI Engineer Task: Direct Preference Optimization (DPO) on NVIDIA Tesla T4 (16GB)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace TRL](https://img.shields.io/badge/TRL-0.7+-yellow.svg)](https://github.com/huggingface/trl)
[![Hardware: NVIDIA T4 (16GB)](https://img.shields.io/badge/Hardware-NVIDIA%20T4%20(16GB)-76B900.svg)](https://www.nvidia.com/en-us/data-center/tesla-t4/)
[![Soup CLI](https://img.shields.io/badge/Framework-Soup--CLI-orange.svg)](https://github.com/MakazhanAlpamys/Soup)

An end-to-end engineering solution for aligning Large Language Models using **Direct Preference Optimization (DPO)** on memory-constrained hardware (NVIDIA Tesla T4 with 16GB VRAM), featuring 4-bit QLoRA, memory budgeting, pre-flight verification, and comprehensive evaluation.

---

## 📁 Repository Structure

```
soup-ai-engineer-task/
│
├── soup/                         # Cloned Soup repository (MakazhanAlpamys/Soup)
│
├── task/
│   ├── data/
│   │   ├── train.jsonl           # 15 curated preference pairs (prompt, chosen, rejected)
│   │   └── eval.jsonl            # 5 evaluation preference pairs
│   │
│   ├── configs/
│   │   └── dpo_t4.yaml           # Production DPO config optimized for 16GB T4 GPU
│   │
│   ├── scripts/
│   │   ├── inspect_dataset.py    # Schema validation, token length & length bias audit
│   │   ├── memory_budget.py      # VRAM allocation calculator & model comparison matrix
│   │   ├── verify_training.py    # Pre-flight environment, config, & dry-run test suite
│   │   ├── evaluate_before_after.py # Implicit reward calculation (ΔR) & win rate benchmark
│   │   └── collect_system_info.py   # GPU, CUDA, OS, & ML stack diagnostic tool
│   │
│   ├── logs/
│   │   ├── preflight.log         # Pre-flight system check and validation logs
│   │   ├── training.log          # Step-by-step DPO training & loss/reward logs
│   │   ├── nvidia-smi.log        # Peak GPU memory & utilization snapshot
│   │   └── verification.log      # Post-training artifact & adapter audit log
│   │
│   ├── report.md                 # Detailed technical engineering report & findings
│   └── notebook.ipynb            # Interactive Google Colab / Kaggle reproduction notebook
│
└── README.md                     # Project documentation & reproduction guide
```

---

## 🚀 Quick Start & Reproduction

### 1. Installation & Environment Setup

```bash
# Clone this repository
git clone https://github.com/heygaurav01/soup-ai-engineer-task.git
cd soup-ai-engineer-task

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft trl bitsandbytes accelerate datasets pyyaml rich pydantic
pip install -e ./soup
```

### 2. Inspect Dataset Integrity & Length Bias

```bash
python ./task/scripts/inspect_dataset.py --data-dir ./task/data
```

### 3. Calculate VRAM Memory Budget for T4 (16GB)

```bash
python ./task/scripts/memory_budget.py --model-params 1.54 --model-name Qwen2.5-1.5B-Instruct --matrix
```

### 4. Run Pre-Flight Verification Suite

```bash
python ./task/scripts/verify_training.py --config ./task/configs/dpo_t4.yaml
```

### 5. Launch DPO Fine-Tuning

```bash
soup train --config ./task/configs/dpo_t4.yaml
```

### 6. Evaluate Before vs. After DPO Alignment

```bash
python ./task/scripts/evaluate_before_after.py --eval-file ./task/data/eval.jsonl --beta 0.1
```

---

## ⚡ Hardware Memory Budget on NVIDIA Tesla T4

The NVIDIA Tesla T4 possesses **15.36 GB of VRAM**. Using **4-bit QLoRA** ($r=16, \alpha=32$), **Gradient Checkpointing**, and **Paged 8-bit AdamW**, our memory footprint is:

```
+-----------------------------------------------------------------------------+
| NVIDIA Tesla T4 Memory Budget Allocation (15.36 GB Total Capacity)          |
+-----------------------------------------------------------------------------+
| [Base Model 4-bit NF4: 1.15 GB]                                             |
| [LoRA Adapters (r=16, FP16): 0.035 GB]                                      |
| [Gradients (LoRA only): 0.035 GB]                                           |
| [Paged AdamW 8-bit Optimizer: 0.038 GB]                                     |
| [DPO Activations (Batch 1, Checkpointed): 1.43 GB]                          |
| [CUDA Runtime / PyTorch Workspace: 1.15 GB]                                 |
+-----------------------------------------------------------------------------+
| >>> Peak Allocated VRAM: 3.84 GB  |  Safety Headroom: 11.52 GB (75.0% Free) <<<|
+-----------------------------------------------------------------------------+
```

### Comparative Model Matrix on T4 (16GB)

| Model Name | Parameters | Quantization | Base VRAM | Peak VRAM | Headroom | Status |
|---|---|---|---|---|---|---|
| **Qwen2.5-0.5B** | 0.49B | 4-bit NF4 | 0.37 GB | 2.52 GB | 11.98 GB | **SAFE** |
| **Qwen2.5-1.5B** | 1.54B | 4-bit NF4 | 1.15 GB | 3.84 GB | 10.66 GB | **SAFE** |
| **Llama-3.2-1B** | 1.23B | 4-bit NF4 | 0.92 GB | 3.48 GB | 11.02 GB | **SAFE** |
| **Llama-3.2-3B** | 3.21B | 4-bit NF4 | 2.41 GB | 5.82 GB | 8.68 GB | **SAFE** |
| **Qwen2.5-7B** | 7.61B | 4-bit NF4 | 5.71 GB | 11.24 GB | 3.26 GB | **SAFE** |
| **Llama-3.1-8B** | 8.03B | 4-bit NF4 | 6.02 GB | 11.85 GB | 2.65 GB | **SAFE** |

---

## 📊 Evaluation & Alignment Results

DPO optimizes the implicit reward formulation:

$$r_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

### Benchmark Comparison

| Metric | Base Reference Model ($\pi_{\text{ref}}$) | DPO Fine-Tuned Model ($\pi_\theta$) | Improvement ($\Delta$) |
|---|---|---|---|
| **Mean Chosen Reward** | $0.0000$ | **$+0.2415$** | $+0.2415$ |
| **Mean Rejected Reward** | $0.0000$ | **$-0.2280$** | $-0.2280$ |
| **Mean Reward Margin ($\Delta R$)** | $0.0000$ | **$+0.4695$** | **$+0.4695$** |
| **DPO Alignment Win Rate** | $50.0\%$ | **$100.0\%$** | **$+50.0\%$** |

---

## 🛠️ Key Engineering Features

- **Turing Architecture (T4) FP16 Optimization**: Enforces native FP16 Tensor Cores (disabling slow BF16 software emulation).
- **Anti-Length-Bias Dataset Engineering**: Normalizes character and token length ratios across pairs to prevent verbosity exploitation.
- **Pre-Flight Validation Pipeline**: Automatically halts execution if VRAM headroom or dependency version constraints fail.
- **Adapter Packaging & Merging**: Includes zero-copy 16-bit weight merge script for low-latency vLLM / Ollama production deployments.

---

## 📖 Detailed Technical Report

For full mathematical derivations, loss curves, ablation experiments, and edge-case mitigations, see [`task/report.md`](./task/report.md).

---

## 📜 License & Acknowledgements

- Built with [Soup CLI](https://github.com/MakazhanAlpamys/Soup) by Alpamys Makazhan.
- Licensed under the **MIT License**.