# Silent Failure Matrix: Direct Preference Optimization & Layer Streaming

This document provides a comprehensive analysis of 15 silent failure modes in LLM Direct Preference Optimization (DPO) fine-tuning and layer streaming architectures on constrained hardware (NVIDIA Tesla T4).

---

### 1. Adapter Not Actually Trainable
- **Failure:** PEFT LoRA matrices injected with `requires_grad=False` or frozen by upstream model wrapping, causing the training loop to run without updating any weights.
- **Detection:** Inspect `[p.requires_grad for p in model.parameters()]` and `model.get_nb_trainable_parameters()` before launching the optimizer.
- **Evidence:** Parameter hash check before vs. after training ($\Delta W \neq 0$) and non-zero Frobenius norm $\|\Delta W\|_F > 0$.
- **Soup Catches It:** Yes, `soup_cli/trainer/dpo.py` asserts `trainable > 0` and prints trainable parameter percentages.
- **What Remains Undetected:** If only a subset of projection matrices are frozen (e.g. `v_proj` trainable while `q_proj` is frozen), no exception is raised.
- **Risk:** High (wasted compute, zero preference learning).

---

### 2. Wrong Dataset Columns / Schema Mismatch
- **Failure:** Preference dataset contains columns like `instruction`, `response`, `output` instead of `prompt`, `chosen`, `rejected`.
- **Detection:** Pre-flight JSON schema validation asserting presence of required keys.
- **Evidence:** `inspect_dataset.py` validates 100% of rows against the required tuple schema.
- **Soup Catches It:** Yes, `soup_cli/data/` rejects datasets missing `chosen` or `rejected` for `task: dpo`.
- **What Remains Undetected:** Swapped columns (e.g., placing the rejected text in the `chosen` field and vice versa) invert preference learning into anti-alignment without raising errors.
- **Risk:** Critical (model learns to produce toxic/dispreferred completions).

---

### 3. Chosen and Rejected Accidentally Identical
- **Failure:** Due to pipeline deduplication or scraping bugs, `chosen == rejected`, yielding $\Delta R = 0$ and zero gradient signal.
- **Detection:** Jaccard similarity and exact string match check between chosen and rejected strings across all rows.
- **Evidence:** `inspect_dataset.py` verifies `identical_chosen_rejected == 0`.
- **Soup Catches It:** Partial (Soup warns on empty strings but does not assert pairwise inequality for every batch row).
- **What Remains Undetected:** Near-identical pairs differing by only 1 whitespace or punctuation mark.
- **Risk:** Medium (dilutes gradient signal and wastes step budget).

---

### 4. Tokenizer / Chat Template Mismatch
- **Failure:** Base model tokenizer uses specific special tokens (`<|im_start|>`, `<|im_end|>`) while dataset formatting injects raw Alpaca or Llama-2 syntax (`[INST]`), causing the model to learn raw strings as text instead of structural boundaries.
- **Detection:** Decode tokenized sequences and check for unmapped special tokens or literal escape sequences.
- **Evidence:** Special token mappings verified via `AutoTokenizer.chat_template` inspection.
- **Soup Catches It:** Yes, `soup_cli/data/chat_templates.py` applies template overrides matching the base model family.
- **What Remains Undetected:** Mismatched system prompt delimiters when merging multi-turn conversations.
- **Risk:** High (model generates raw delimiter tokens in live chat).

---

### 5. Padding and Truncation Right-Side Loss Masking
- **Failure:** In DPO, if sequences are left-padded incorrectly or if the prompt is truncated from the wrong end, the target response tokens are clipped or masked from the log-likelihood calculation.
- **Detection:** Verify that padding is set to `right` for causal LM training and prompt length is capped independently (`max_prompt_length`).
- **Evidence:** Sequence length audit showing $L_{\text{pair}} \le 1024$ and prompt lengths $< 512$.
- **Soup Catches It:** Yes, `soup_cli/trainer/_trl_compat.py` enforces sequence bounds and sets tokenizer padding rules.
- **What Remains Undetected:** Sub-token truncation occurring inside multi-byte UTF-8 Russian characters.
- **Risk:** High (corrupted vocabulary tokens and grammatical degeneration).

---

### 6. Reference Model Synchronization & Weight Divergence
- **Failure:** In standard DPO, the reference model $\pi_{\text{ref}}$ must remain frozen. If the optimizer accidentally updates $\pi_{\text{ref}}$, implicit reward $r_\theta = \beta (\log \pi_\theta - \log \pi_{\text{ref}}) \to 0$, causing training collapse.
- **Detection:** Verify reference model parameter gradients (`p.grad is None`) and evaluate log-probability divergence.
- **Evidence:** Reference model shares frozen base weights with adapter disabling during reference forward.
- **Soup Catches It:** Yes, `DPOTrainer` wraps reference evaluation with `torch.no_grad()`.
- **What Remains Undetected:** Shared memory mutations if reference adapters are instantiated on meta skeletons.
- **Risk:** Critical (complete failure of policy optimization).

---

### 7. Incorrect DPO $\beta$ (Beta) Scaling
- **Failure:** $\beta$ set too low ($\le 0.001$) leads to aggressive KL drift and garbage output; $\beta$ set too high ($\ge 2.0$) freezes the policy, preventing learning.
- **Detection:** Pre-flight configuration range validation ($0.01 \le \beta \le 0.5$).
- **Evidence:** Config validated with $\beta = 0.1$, yielding stable reward margin growth $+0.4695$.
- **Soup Catches It:** Yes, Soup validates `dpo_beta` in Pydantic schema (`schema.py`).
- **What Remains Undetected:** Domain-dependent $\beta$ sensitivity (e.g., code tasks often require $\beta=0.05$ while chat safety requires $\beta=0.2$).
- **Risk:** High (degenerated generation quality).

---

### 8. Optimizer Not Updating (Learning Rate = 0 or Frozen States)
- **Failure:** LR scheduler warmdown drops LR to 0 prematurely, or optimizer step is skipped due to infinite/NaN gradient unscaling.
- **Detection:** Track step count, learning rate per step, and verify parameter delta norm $\|\Delta W\| > 0$.
- **Evidence:** `verification.log` verifies parameter updates occurred across 392 adapter weight tensors.
- **Soup Catches It:** Yes, `SoupTrainerCallback` monitors optimizer steps.
- **What Remains Undetected:** Optimizer updates that underflow FP16 numerical precision without triggering NaN.
- **Risk:** High (silent no-op training).

---

### 9. Zero or Exploding Gradient Norms
- **Failure:** Vanishing gradients ($\|\nabla W\| < 10^{-7}$) or exploding gradients ($\|\nabla W\| > 100$) due to improper loss scaling.
- **Detection:** Log and assert gradient norms ($0.1 \le \|\nabla W\| \le 5.0$) at every logging step.
- **Evidence:** `training.log` shows smooth gradient norms between $0.228$ and $0.421$.
- **Soup Catches It:** Yes, TRL logs `grad_norm` at each logging step.
- **What Remains Undetected:** Gradient norm computed only over a subset of distributed ranks.
- **Risk:** High (instability or stalling).

---

### 10. Layer Streaming Desynchronization & Cache Pollution
- **Failure:** When streaming layers from host RAM/disk, a prefetch buffer is overwritten before the backward pass completes its computation, causing stale layer weights to compute gradients.
- **Detection:** Parity test comparing streamed vs. resident forward/backward passes on deterministic inputs.
- **Evidence:** `StreamingSetupMixin` uses dedicated CUDA streams with double buffering and explicit synchronization events.
- **Soup Catches It:** Yes, Soup implements `_STREAM_REFILL_BEFORE_BACKWARD = True` and save guards.
- **What Remains Undetected:** Minor host-to-device PCIe transfer jitter under heavy CPU memory load.
- **Risk:** Critical (corrupted gradients and incorrect model weights).

---

### 11. Quantization / BitsAndBytes Precision Mismatch
- **Failure:** Quantizing in 4-bit NF4 without double quant, or attempting BF16 training on Turing T4 GPUs lacking native BF16 arithmetic units.
- **Detection:** Pre-flight GPU compute capability probe (`cc == (7, 5)` $\to$ force FP16).
- **Evidence:** `collect_system_info.py` and `dpo_t4.yaml` enforce `fp16: true`.
- **Soup Catches It:** Yes, `soup_cli/utils/gpu.py` forces `fp16=True, bf16=False` on Turing architectures.
- **What Remains Undetected:** Quantization rounding noise on outlier weights in large attention heads.
- **Risk:** Medium (2-3x training slowdown if BF16 software emulation is accidentally invoked).

---

### 12. Checkpoint and Artifact Save Inconsistency
- **Failure:** Training completes successfully in memory, but disk save fails silently or writes an empty `adapter_model.safetensors` due to uncommitted buffers.
- **Detection:** Post-training file existence, safetensors header inspection, and non-empty byte count assertion.
- **Evidence:** `verification.log` confirms `adapter_model.safetensors` (73.66 MB) and valid config files.
- **Soup Catches It:** Yes, `assert_streamed_adapter_saved` inspects adapter output on disk.
- **What Remains Undetected:** Checkpoint saved on a remote network share that disconnects during export.
- **Risk:** Critical (loss of entire trained artifact).

---

### 13. Loss Decreasing Without Meaningful Parameter Updates
- **Failure:** DPO loss decreases merely because the model increases the sequence log-likelihood of all tokens indiscriminately, rather than increasing the relative margin between chosen and rejected.
- **Detection:** Track implicit reward margin $\Delta R = R(y_w) - R(y_l)$ and chosen vs. rejected log-probability trajectories.
- **Evidence:** `training.log` shows chosen reward increasing ($+0.2468$) while rejected reward decreases ($-0.2315$), proving true preference separation.
- **Soup Catches It:** Yes, TRL logs `rewards/margins` and `rewards/accuracies`.
- **What Remains Undetected:** Length exploitation where the model increases reward purely by predicting longer repetitive sentences.
- **Risk:** High (pseudo-alignment and length hacking).

---

### 14. GPU Memory Measurement Mismatch (Allocated vs. Reserved vs. OS)
- **Failure:** PyTorch reports `torch.cuda.memory_allocated() = 3.2 GB`, but the GPU crashes with OOM because `nvidia-smi` sees 15.3 GB due to memory fragmentation and cached allocator blocks.
- **Detection:** Measure both `memory_allocated()`, `memory_reserved()`, and query `nvidia-smi` during peak forward-backward passes.
- **Evidence:** Memory budget calculator and `nvidia-smi.log` reconcile allocated (3.84 GB) against reserved limits.
- **Soup Catches It:** Yes, Soup's VRAM pre-flight charges activation and allocator overhead.
- **What Remains Undetected:** External processes (e.g. Xorg, desktop window managers) consuming unmonitored VRAM.
- **Risk:** High (unexpected CUDA OOM crashes).

---

### 15. Reported Throughput (tok/s) Not Representing Actual Training Work
- **Failure:** Benchmark logs report high tokens/second by counting padding tokens or skipped batches.
- **Detection:** Calculate effective throughput using non-padded token counts and actual wall-clock elapsed step time:
  $$\text{Effective Throughput} = \frac{\sum \text{unpadded tokens}}{\Delta t_{\text{step}}}$$
- **Evidence:** `training.log` reports 3.65 samples/sec on batch size 1 with 8 gradient accumulation steps on T4.
- **Soup Catches It:** Partial (logs standard HF Trainer samples/sec).
- **What Remains Undetected:** Throughput inflated by sequences padded to `max_length`.
- **Risk:** Low (reporting inaccuracy without affecting model weights).
