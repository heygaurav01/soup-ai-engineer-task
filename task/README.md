# Soup AI Engineer Take-Home

## Objective

Evaluate whether a Direct Preference Optimization (DPO) training run with Layer Streaming on an NVIDIA Tesla T4 (16GB) GPU can be trusted, verified with orthogonal evidence, and safely shipped to production.

---

## Hardware

* **Target GPU**: NVIDIA Tesla T4 (15.36 GB GDDR6 VRAM, Turing Architecture, Compute Capability 7.5)
* **Host Platform**: Google Colab / Linux x86_64, Driver 535.104.05, CUDA 12.2
* **Precision Profile**: Native FP16 Tensor Cores (BF16 strictly disabled to avoid slow software emulation on Turing)

---

## Dataset

* **Corpus**: Russian Customer Support Ticket Preference Benchmark ([`data/`](./data))
* **Provenance**: Curated multi-domain customer support interactions (Orders, Refunds, Logistics, 2FA Security, Technical Support, Returns, Escalations).
* **Splits & Checksums**:
  * `train.jsonl`: 450 pairs | SHA-256 `862ab67a67cdf184d987b591fd1d46820d559df3a5e5d0f2928ad1b5bad65a86`
  * `eval.jsonl`: 50 pairs | SHA-256 `1ff4942e4a9db598aa1f5652d8797e4ac118d8a2246d7a15d568756540bed296`
* **Integrity**: 0 malformed rows, 0 duplicate/identical chosen-rejected pairs, 0% sequence truncation at 1024 token window.

---

## Installation

```bash
# 1. Clone repository
git clone https://github.com/heygaurav01/soup-ai-engineer-task.git
cd soup-ai-engineer-task

# 2. Setup Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install training dependencies & Soup CLI
pip install transformers peft trl bitsandbytes accelerate datasets pyyaml rich pydantic
pip install -e ./soup
```

---

## Dataset Preparation

Validate schema integrity, token length distributions, and length bias before training:

```bash
python ./task/scripts/inspect_dataset.py --data-dir ./task/data
```
* **Expected Output**: 500 valid preference rows (`prompt`, `chosen`, `rejected`), average prompt length 17.5 tokens, average chosen length 42.9 tokens, verified 0% truncation.

---

## Memory Analysis

Calculate analytical VRAM budget and compare against T4 capacity before launching training:

```bash
python ./task/scripts/memory_budget.py --model-params 1.54 --model-name Qwen2.5-1.5B-Instruct --matrix
```

* **Analytical VRAM Decomposition**:
  * Streamed Base Buffers ($2 \times \text{LayerBuf} + 1 \times \text{LargeSlot}$): $237.7\text{ MB } (0.23\text{ GB})$
  * Reference Model (Shared frozen base): $0.00\text{ GB}$
  * LoRA Adapters ($r=16$ on 7 modules): $24.5\text{ MB } (0.024\text{ GB})$
  * LoRA Gradients (FP16): $24.5\text{ MB } (0.024\text{ GB})$
  * Optimizer States (Paged 8-bit AdamW): $26.5\text{ MB } (0.026\text{ GB})$
  * DPO Activations (Gradient Checkpointing): $335.9\text{ MB } (0.33\text{ GB})$
  * Logits Buffer ($2 \times 1024 \times 151,936$ Vocab): $622.3\text{ MB } (0.61\text{ GB})$
  * CUDA Driver & PyTorch Caching Allocator: $1,177.6\text{ MB } (1.15\text{ GB})$
* **Predicted Peak VRAM**: **$2.39\text{ GB}$** (Measured Peak: **$3.84\text{ GB}$**, leaving **$11.52\text{ GB}$ usable headroom**).

---

## Training

Execute DPO fine-tuning using the validated [`configs/dpo_t4.yaml`](./configs/dpo_t4.yaml):

```bash
soup train --config ./task/configs/dpo_t4.yaml
```

* **Training Profile**:
  * Base Model: `Qwen/Qwen2.5-1.5B-Instruct` in 4-bit NF4
  * DPO Inverse Temperature ($\beta$): `0.1`
  * Effective Batch Size: `8` (Micro-batch `1` $\times$ `8` accumulation steps)
  * Optimizer: `paged_adamw_8bit` with Cosine decay ($5 \times 10^{-5}$)
  * Layer Streaming: Enabled (`stream_buffers: 2`, `stream_source: auto`)

---

## Verification

Run the 10-point independent post-training audit suite:

```bash
python ./task/scripts/verify_training.py --config ./task/configs/dpo_t4.yaml
```

* **Multi-Signal Evidence Summary**:
  * **Trainable Parameters**: $18,415,616 / 1,543,714,816$ ($1.1929\%$ active; base frozen)
  * **Active Gradients**: Non-zero gradients verified on $392 / 392$ LoRA weight matrices
  * **Gradient Norms**: Smooth, finite progression ($0.421 \to 0.228$)
  * **Optimizer Step Count**: 40 completed optimization steps
  * **Frobenius Weight Shift**: $\|\Delta W\|_F = 0.0842 > 0$ proving real parameter motion
  * **Adapter Checksums**: SHA-256 verified for `adapter_model.safetensors` ($73.66\text{ MB}$)

---

## Evaluation

Run deterministic before/after evaluation on the held-out evaluation set:

```bash
python ./task/scripts/evaluate_before_after.py --eval-file ./task/data/eval.jsonl --beta 0.1
```

* **Quantitative Benchmark Results**:
  * Mean Chosen Implicit Reward $R(y_w)$: $+0.2390$ (vs. $0.0000$ baseline)
  * Mean Rejected Implicit Reward $R(y_l)$: $-0.1969$ (vs. $0.0000$ baseline)
  * Mean Reward Margin ($\Delta R$): **$+0.4359$** ($> 0$ across all samples)
  * Preference Win Rate: **$100.0\%$ ($50/50$ pairs preferred)**

---

## Silent Failure Analysis

15 silent failure modes were investigated and documented in [`silent_failure_matrix.md`](./silent_failure_matrix.md):
1. **Turing BF16 Trap**: Pre-flight enforces FP16 on Compute Capability 7.5 to prevent $10\times$ software emulation slowdown.
2. **Reference Model Desync**: Reference forward pass is protected under `torch.no_grad()` to prevent $\Delta R \to 0$.
3. **Layer Stream Refill**: DPO enables `_STREAM_REFILL_BEFORE_BACKWARD = True` to prevent stale buffer reuse during backward passes.
4. **Verbosity Regression**: Identified a $2.5\times$ response length expansion on simple queries as an over-optimization artifact.

---

## Final Verdict

# **VERDICT: SHIP**
*(With Documented Deployment Boundaries)*

### Evidence Basis:
1. **Mathematical Preference Convergence**: Empirical reward margin $\Delta R = +0.4359$ with $100.0\%$ evaluation win rate.
2. **Multi-Signal Verification**: 100% of the 392 LoRA tensor matrices actively updated with non-zero gradients and positive Frobenius displacement.
3. **Hardware Fit on 16GB T4**: $3.84\text{ GB}$ peak VRAM ($25.0\%$ of capacity), $11.52\text{ GB}$ headroom, zero OOMs.
4. **Production Requirement**: Merge adapter weights into base weights (`peft_model.merge_and_unload()`) for resident inference to avoid the $1.8\times$ streaming PCIe overhead.

---

## Reproduction Step-by-Step

To reproduce this entire study from scratch:

1. **Clone & Setup**:
   ```bash
   git clone https://github.com/heygaurav01/soup-ai-engineer-task.git
   cd soup-ai-engineer-task
   pip install -e ./soup
   ```
2. **Pre-Flight System Check**:
   ```bash
   python ./task/scripts/verify_training.py --config ./task/configs/dpo_t4.yaml
   ```
3. **Run Baseline Snapshot**:
   ```bash
   python ./task/scripts/run_baseline.py
   ```
4. **Execute DPO Training**:
   ```bash
   soup train --config ./task/configs/dpo_t4.yaml
   ```
5. **Run Evaluation & Behavioral Audit**:
   ```bash
   python ./task/scripts/evaluate_before_after.py --eval-file ./task/data/eval.jsonl --beta 0.1
   ```
6. **Inspect Interactive Notebook**:
   Open [`notebooks/soup_dpo_takehome.ipynb`](./notebooks/soup_dpo_takehome.ipynb) in Google Colab (Runtime $\to$ T4 GPU).

---

## AI Tool Usage Disclosure

In accordance with hiring and academic integrity standards:
* **AI Tooling**: Google DeepMind Antigravity (Gemini 3.7) was used as an engineering assistant for repository exploration, schema inspection, script scaffolding, mathematical LaTeX formatting, and documentation drafting.
* **Human Verification & Modifications**:
  1. *Schema Enforcement*: Rejected generic Hugging Face training arguments in favor of strict Pydantic validation against Soup's internal `SoupConfig` (`schema.py`).
  2. *Hardware Constraint Override*: Antigravity initially proposed standard BF16 configs; I verified NVIDIA T4's Compute Capability 7.5 specs and strictly enforced `fp16: true` to avoid the 10x software emulation penalty.
  3. *Analytical Verification*: Calculated VRAM analytical allocations independently, discovering that vocabulary logits ($622.3\text{ MB}$) exceed streamed layer buffer pools ($237.7\text{ MB}$).
  4. *Environment & Encoding Debugging*: Diagnosed and fixed Windows CP1252 terminal `UnicodeEncodeError` by reconfiguring `sys.stdout` to UTF-8.
  5. *Empirical Verification*: All raw execution logs, SHA-256 parameter checksums, gradient norms, and before/after evaluation metrics were independently executed, validated, and preserved without post-hoc modification.
