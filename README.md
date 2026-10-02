# 🍲 Soup AI Engineer Take-Home: DPO & Layer Streaming on NVIDIA Tesla T4 (16GB)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace TRL](https://img.shields.io/badge/TRL-0.7+-yellow.svg)](https://github.com/huggingface/trl)
[![Hardware: NVIDIA T4 (16GB)](https://img.shields.io/badge/Hardware-NVIDIA%20T4%20(16GB)-76B900.svg)](https://www.nvidia.com/en-us/data-center/tesla-t4/)
[![Soup CLI](https://img.shields.io/badge/Framework-Soup--CLI-orange.svg)](https://github.com/MakazhanAlpamys/Soup)

An end-to-end engineering solution for aligning Large Language Models using **Direct Preference Optimization (DPO)** with **Layer Streaming** on hardware-constrained infrastructure (NVIDIA Tesla T4 with 16GB VRAM), featuring 4-bit QLoRA, analytical memory budgeting, pre-flight verification, and comprehensive behavioral evaluation.

---

## 📁 Repository Structure

```text
soup-ai-engineer-takehome/
│
├── README.md                     # Main repository guide & reproduction walkthrough
├── report.md                     # Concise <= 2-page scientific & technical report (SHIP verdict)
├── silent_failure_matrix.md      # Comprehensive 15-point silent failure analysis & audit
│
├── configs/
│   └── dpo_t4.yaml               # Validated SoupConfig schema for DPO on 16GB T4 GPU
│
├── data/
│   ├── README.md                 # Dataset provenance, category breakdown & SHA-256 manifest
│   ├── train.jsonl               # 450 pairwise preference tuples (prompt, chosen, rejected)
│   └── eval.jsonl                # 50 held-out evaluation pairs
│
├── scripts/
│   ├── collect_system_info.py    # GPU, CUDA, OS, & software stack diagnostic tool
│   ├── inspect_dataset.py        # Schema validation, token length & length bias audit
│   ├── memory_budget.py          # Analytical VRAM allocation calculator & model matrix
│   ├── verify_training.py        # 10-point pre-flight & post-training verification suite
│   ├── run_baseline.py           # Reproducible unaligned baseline snapshot runner
│   └── evaluate_before_after.py  # Implicit reward (ΔR), win-rate, & regression evaluator
│
├── notebooks/
│   └── soup_dpo_takehome.ipynb   # Interactive Google Colab / T4 reproduction notebook
│
└── logs/
    ├── environment.log           # Hardware diagnostics, CUDA version, pinned packages
    ├── nvidia-smi.log            # Peak GPU memory & driver utilization snapshot
    ├── dataset.log               # Dataset token length distributions & schema audit
    ├── preflight.log             # Pre-flight checklist and dry-run execution
    ├── training.log              # Raw DPO training steps, gradient norms, & loss
    └── verification.log          # 10-point independent post-training audit trail
```

---

## 🚀 Quick Start & Reproduction Walkthrough

### 1. Installation & Environment Setup

```bash
# Clone this repository
git clone https://github.com/heygaurav01/soup-ai-engineer-task.git
cd soup-ai-engineer-task

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install training dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft trl bitsandbytes accelerate datasets pyyaml rich pydantic
pip install -e ./soup
```

### 2. Inspect Dataset Integrity & Token Length Bias

```bash
python ./task/scripts/inspect_dataset.py --data-dir ./task/data
```
* **Output**: Validates 500 rows, 0 malformed rows, 0 duplicate chosen/rejected pairs, 0% sequence truncation.

### 3. Calculate Analytical VRAM Memory Budget for T4 (16GB)

```bash
python ./task/scripts/memory_budget.py --model-params 1.54 --model-name Qwen2.5-1.5B-Instruct --matrix
```
* **Output**: Decomposes base layer buffers ($237.7\text{ MB}$), LoRA adapters ($24.5\text{ MB}$), gradients ($24.5\text{ MB}$), optimizer ($26.5\text{ MB}$), activations ($335.9\text{ MB}$), and logits projection ($622.3\text{ MB}$). Predicts peak VRAM at $\approx 2.39\text{ GB}$ (leaving $>11.5\text{ GB}$ headroom).

### 4. Run Pre-Flight & Post-Training Verification Suite

```bash
python ./task/scripts/verify_training.py --config ./task/configs/dpo_t4.yaml
```

### 5. Launch DPO Fine-Tuning with Layer Streaming

```bash
soup train --config ./task/configs/dpo_t4.yaml
```

### 6. Evaluate Before vs. After DPO Alignment

```bash
python ./task/scripts/evaluate_before_after.py --eval-file ./task/data/eval.jsonl --beta 0.1
```
* **Output**: Computes implicit rewards $R(y_w), R(y_l)$, reward margin ($\Delta R = +0.4359$), win rate ($100.0\%$), and analyzes length/verbosity regressions.

---

## 📊 Summary of Experimental Results

| Metric | Target / Baseline | DPO Fine-Tuned Policy | Status |
| :--- | :--- | :--- | :--- |
| **Peak VRAM on T4 (15.36 GB)** | $\le 14.5\text{ GB}$ | **$3.84\text{ GB}$ ($25.0\%$ of VRAM)** | **SAFE ($11.52\text{ GB}$ Headroom)** |
| **Trainable Parameters** | Low-rank adapter ($r=16$) | $18,415,616$ ($1.1929\%$ of base) | **Exact LoRA Coverage** |
| **Gradients Active** | 100% of LoRA matrices | **$392 / 392$ tensors active** | **Zero Leaked Gradients** |
| **Frobenius Weight Shift ($\|\Delta W\|_F$)** | $> 0$ | **$0.0842$** | **Confirmed Parameter Updates** |
| **DPO Loss Convergence** | Initial $\to$ Final | $0.6931 \to 0.4085$ | **Monotonic Convergence** |
| **Implicit Reward Margin ($\Delta R$)** | $> +0.3000$ | **$\mathbf{+0.4359}$** | **Strong Preference Separation** |
| **Evaluation Set Win Rate** | $\ge 85.0\%$ | **$\mathbf{100.0\%}$ ($50 / 50$ pairs)** | **Robust Behavioral Alignment** |

---

## 🛡️ Rigorous 10-Point Verification Summary

1. **Trainable Parameter Selection**: $18.4\text{M}$ trainable parameters ($1.19\%$), base weights frozen.
2. **Gradient Existence**: Non-zero gradients verified on all $392$ LoRA matrices ($0$ on base model).
3. **Gradient Norms**: Finite, non-vanishing norms ($0.421 \to 0.228$).
4. **Optimizer Updates**: 40 completed steps in Paged AdamW 8-bit with updated first/second momenta.
5. **Adapter Serialization**: Valid `adapter_model.safetensors` ($73.66\text{ MB}$) and `adapter_config.json`.
6. **Parameter Checksums**: Cryptographic SHA-256 hashes generated for all serialized artifacts.
7. **Weight Delta Frobenius Norm**: $\|\Delta W\|_F = 0.0842 > 0$ proving real parameter motion.
8. **Changed Trainable Tensors**: $100.0\%$ ($392/392$) of LoRA matrices updated; $0\%$ base weights altered.
9. **Checkpoint Metadata**: Valid lineage recorded (`Qwen2.5-1.5B-Instruct`, $r=16, \alpha=32$).
10. **Before / After Behavior**: Ground truth evaluation on held-out test data showing $\Delta R = +0.4359$.

---

## 📑 Core Documentation Links

* [**Final Technical Report (report.md)**](./task/report.md): Concise $\le 2$-page technical report detailing the environment, memory decomposition, training evidence, and **SHIP** deployment verdict.
* [**Silent Failure Matrix (silent_failure_matrix.md)**](./task/silent_failure_matrix.md): In-depth audit of 15 silent failure modes in DPO and layer streaming.
* [**Dataset Documentation (data/README.md)**](./task/data/README.md): Detailed ticket categories, length distributions, and SHA-256 verification hashes.