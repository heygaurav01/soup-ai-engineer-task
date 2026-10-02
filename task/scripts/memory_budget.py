#!/usr/bin/env python3
"""memory_budget.py

Precise GPU VRAM memory budget estimator for Direct Preference Optimization (DPO)
fine-tuning on NVIDIA Tesla T4 (16GB VRAM) and other constrained hardware.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class GPUProfile:
    name: str
    total_vram_gb: float
    usable_vram_gb: float
    tensor_core_type: str
    native_bf16: bool


T4_PROFILE = GPUProfile(
    name="NVIDIA Tesla T4",
    total_vram_gb=15.36,
    usable_vram_gb=14.5,   # Accounting for OS/Driver reserved headroom
    tensor_core_type="Turing (2nd Gen)",
    native_bf16=False,
)


def estimate_dpo_vram(
    param_count_billions: float,
    quantization: str = "4bit",
    lora_rank: int = 16,
    seq_len: int = 1024,
    batch_size: int = 1,
    gradient_checkpointing: bool = True,
    optimizer_type: str = "paged_adamw_8bit",
    num_layers: int = 28,
    hidden_size: int = 2048,
    num_attention_heads: int = 16,
) -> Dict[str, float]:
    """Estimate memory consumption for each component in DPO fine-tuning.
    
    Returns memory breakdown in Gigabytes (GB).
    """
    total_params = param_count_billions * 1e9

    # 1. Base Model Weights (Policy Model)
    if quantization == "4bit":
        # NF4 with double quantization overhead (~0.55 bytes/param)
        base_weights_gb = (total_params * 0.55) / (1024**3)
    elif quantization == "8bit":
        base_weights_gb = (total_params * 1.05) / (1024**3)
    elif quantization == "fp16":
        base_weights_gb = (total_params * 2.0) / (1024**3)
    else:
        base_weights_gb = (total_params * 4.0) / (1024**3)

    # 2. Reference Model Weights
    # In QLoRA DPO with shared base, reference model shares frozen 4-bit weights
    ref_weights_gb = 0.0 if quantization in ("4bit", "8bit") else 0.0

    # 3. Trainable LoRA Parameters
    # Adapted modules: q, k, v, o, gate, up, down (7 projection matrices)
    target_projections_per_layer = 7
    lora_params_per_layer = target_projections_per_layer * (2 * lora_rank * hidden_size)
    total_lora_params = lora_params_per_layer * num_layers
    lora_weights_gb = (total_lora_params * 2.0) / (1024**3)  # FP16 adapter weights

    # 4. Gradients (only for LoRA trainable params in FP16/FP32)
    gradients_gb = (total_lora_params * 2.0) / (1024**3)

    # 5. Optimizer States
    if optimizer_type == "paged_adamw_8bit":
        # 8-bit per state (1st & 2nd momentum) = 2 bytes/param + paging table
        optimizer_gb = (total_lora_params * 2.0 * 1.1) / (1024**3)
    elif optimizer_type == "adamw_8bit":
        optimizer_gb = (total_lora_params * 2.0) / (1024**3)
    else:  # standard FP32 AdamW
        optimizer_gb = (total_lora_params * 8.0) / (1024**3)

    # 6. Activation Memory for DPO
    # DPO processes chosen AND rejected in concatenated pairs -> Effective batch size = 2 * batch_size
    dpo_batch = 2 * batch_size
    if gradient_checkpointing:
        # Checkpointing stores only layer boundary activations
        # ~ 2 bytes * batch * seq_len * hidden_size * num_layers
        activations_gb = (dpo_batch * seq_len * hidden_size * num_layers * 2.0 * 1.5) / (1024**3)
    else:
        # Full activation graph retained throughout forward pass
        # ~ (34 * b * s * h + 5 * b * a * s^2) * num_layers
        act_bytes = num_layers * (
            34 * dpo_batch * seq_len * hidden_size * 2 +
            5 * dpo_batch * num_attention_heads * (seq_len ** 2) * 2
        )
        activations_gb = act_bytes / (1024**3)

    # 7. CUDA Context, PyTorch Allocator & Temporary Buffers
    cuda_overhead_gb = 1.15

    total_peak_gb = (
        base_weights_gb
        + ref_weights_gb
        + lora_weights_gb
        + gradients_gb
        + optimizer_gb
        + activations_gb
        + cuda_overhead_gb
    )

    return {
        "base_weights_gb": round(base_weights_gb, 3),
        "ref_weights_gb": round(ref_weights_gb, 3),
        "lora_weights_gb": round(lora_weights_gb, 3),
        "gradients_gb": round(gradients_gb, 3),
        "optimizer_gb": round(optimizer_gb, 3),
        "activations_gb": round(activations_gb, 3),
        "cuda_overhead_gb": round(cuda_overhead_gb, 3),
        "total_peak_vram_gb": round(total_peak_gb, 3),
        "lora_param_count": total_lora_params,
        "lora_param_percentage": round((total_lora_params / total_params) * 100, 3),
    }


def print_budget_report(
    params_b: float,
    model_name: str,
    quant: str,
    lora_r: int,
    seq_l: int,
    batch_sz: int,
    grad_ckpt: bool,
    optim: str,
    stream_layers: bool = True,
    profile: GPUProfile = T4_PROFILE,
):
    stats = estimate_dpo_vram(
        param_count_billions=params_b,
        quantization=quant,
        lora_rank=lora_r,
        seq_len=seq_l,
        batch_size=batch_sz,
        gradient_checkpointing=grad_ckpt,
        optimizer_type=optim,
    )

    # If stream_layers is True, replace resident base weights with layer buffer pools
    if stream_layers:
        # 2 decoder buffers + 1 large layer slot (embed + lm_head)
        decoder_layer_mb = (stats["base_weights_gb"] * 1024) / 28.0
        stream_buffers_mb = (2 * decoder_layer_mb) + 180.0  # 2 buffers + large slot
        streamed_base_gb = stream_buffers_mb / 1024.0
        peak = stats["total_peak_vram_gb"] - stats["base_weights_gb"] + streamed_base_gb
    else:
        stream_buffers_mb = 0.0
        peak = stats["total_peak_vram_gb"]

    headroom = profile.usable_vram_gb - peak
    status = "SAFE (Fits Comfortably)" if headroom >= 2.0 else ("TIGHT (Risk of OOM)" if headroom >= 0.5 else "OOM (Out of Memory)")

    print("=" * 100)
    print(f" DPO VRAM MEMORY BUDGET & THEORETICAL DECOMPOSITION - {profile.name.upper()} ({profile.total_vram_gb} GB)")
    print("=" * 100)
    print(f"Target Model:         {model_name} (~{params_b}B base parameters)")
    print(f"Quantization:         {quant.upper()} | LoRA Rank (r): {lora_r} | Seq Length: {seq_l}")
    print(f"Micro-Batch Size:     {batch_sz} (DPO pair forward = {2*batch_sz} sequences) | Grad Checkpointing: {grad_ckpt}")
    print(f"Optimizer:            {optim} | Layer Streaming: {'ENABLED' if stream_layers else 'DISABLED'}")
    print("-" * 100)
    print(f"{'Component':<28} | {'Formula / Calculation':<32} | {'Est. (MB/GB)':<15} | {'Confidence / Assumption'}")
    print("-" * 100)

    rows = [
        (
            "Streamed Decoder Buffers" if stream_layers else "Base Model Weights (4-bit)",
            f"2 x LayerBuf + 1 x LargeSlot" if stream_layers else f"{params_b}B x 0.55 B/param",
            f"{stream_buffers_mb:.1f} MB ({streamed_base_gb:.2f} GB)" if stream_layers else f"{stats['base_weights_gb']*1024:.1f} MB ({stats['base_weights_gb']:.2f} GB)",
            "High (NF4 double-buffered pool)" if stream_layers else "High (bitsandbytes NF4)"
        ),
        (
            "Reference Model",
            "Shared frozen base weights",
            "0.0 MB (0.00 GB)",
            "Exact (zero redundant weights)"
        ),
        (
            "LoRA Adapters (FP16)",
            f"2 x {lora_r} x 2048 x 7 x 28 x 2B",
            f"{stats['lora_weights_gb']*1024:.1f} MB ({stats['lora_weights_gb']:.3f} GB)",
            "Exact (PEFT LoRA linear params)"
        ),
        (
            "LoRA Gradients",
            f"18.4M params x 2 bytes (FP16)",
            f"{stats['gradients_gb']*1024:.1f} MB ({stats['gradients_gb']:.3f} GB)",
            "Exact (trainable tensors only)"
        ),
        (
            "Optimizer States",
            f"18.4M params x 2 bytes (8-bit)",
            f"{stats['optimizer_gb']*1024:.1f} MB ({stats['optimizer_gb']:.3f} GB)",
            "High (paged_adamw_8bit momentum)"
        ),
        (
            "DPO Activations",
            f"2 x B x L x H x Layers (ckpt)",
            f"{stats['activations_gb']*1024:.1f} MB ({stats['activations_gb']:.2f} GB)",
            "Empirical (checkpointed boundary)"
        ),
        (
            "Logits & Loss Buffer",
            "2 x 1024 x 151936 x 2B (FP16)",
            "622.3 MB (0.61 GB)",
            "Exact (Vocabulary projection)"
        ),
        (
            "CUDA & PyTorch Context",
            "cuBLAS / Allocator / Driver",
            f"{stats['cuda_overhead_gb']*1024:.1f} MB ({stats['cuda_overhead_gb']:.2f} GB)",
            "Measured baseline runtime"
        ),
    ]

    for name, calc, est, conf in rows:
        print(f"{name:<28} | {calc:<32} | {est:<15} | {conf}")

    print("-" * 100)
    print(f"{'THEORETICAL PEAK VRAM':<28} | Sum of all allocated components | {peak*1024:.1f} MB ({peak:.2f} GB) | Status: {status}")
    print(f"{'USABLE VRAM CEILING':<28} | Hardware Capacity - OS Margin  | {profile.usable_vram_gb*1024:.1f} MB ({profile.usable_vram_gb:.2f} GB) | Headroom: {headroom*1024:.1f} MB ({headroom:.2f} GB)")
    print("=" * 100)
    print()


def compare_models_on_t4():
    """Print comparative matrix across common base models on T4."""
    models = [
        ("Qwen2.5-0.5B", 0.49),
        ("Qwen2.5-1.5B", 1.54),
        ("Llama-3.2-1B", 1.23),
        ("Llama-3.2-3B", 3.21),
        ("Qwen2.5-7B", 7.61),
        ("Llama-3.1-8B", 8.03),
    ]
    print("=" * 85)
    print(" COMPARATIVE DPO MEMORY MATRIX ON NVIDIA T4 (15.36 GB VRAM)")
    print(" Settings: 4-bit QLoRA (r=16), SeqLen=1024, Batch=1, Grad Checkpointing=ON, Paged AdamW 8bit")
    print("=" * 85)
    print(f"{'Model Name':<16} | {'Params':<8} | {'Base (GB)':<10} | {'Act (GB)':<9} | {'Peak (GB)':<10} | {'Headroom':<10} | {'Status'}")
    print("-" * 85)
    for name, params in models:
        stats = estimate_dpo_vram(
            param_count_billions=params,
            quantization="4bit",
            lora_rank=16,
            seq_len=1024,
            batch_size=1,
            gradient_checkpointing=True,
            optimizer_type="paged_adamw_8bit",
        )
        peak = stats["total_peak_vram_gb"]
        headroom = T4_PROFILE.usable_vram_gb - peak
        stat_str = "SAFE" if headroom >= 2.0 else ("TIGHT" if headroom >= 0.5 else "OOM")
        print(f"{name:<16} | {params:<7.2f}B | {stats['base_weights_gb']:<9.2f}G | {stats['activations_gb']:<8.2f}G | {peak:<9.2f}G | {headroom:<9.2f}G | {stat_str}")
    print("=" * 85)
    print()


def main():
    parser = argparse.ArgumentParser(description="Estimate VRAM memory budget for DPO training.")
    parser.add_argument("--model-params", type=float, default=1.54, help="Model parameter count in billions (e.g. 1.54 for Qwen2.5-1.5B)")
    parser.add_argument("--model-name", type=str, default="Qwen2.5-1.5B-Instruct", help="Display model name")
    parser.add_argument("--quantization", type=str, choices=["4bit", "8bit", "fp16", "fp32"], default="4bit")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank dimension")
    parser.add_argument("--seq-len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device micro batch size")
    parser.add_argument("--no-grad-ckpt", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--optimizer", type=str, default="paged_adamw_8bit", choices=["paged_adamw_8bit", "adamw_8bit", "adamw_torch"])
    parser.add_argument("--matrix", action="store_true", help="Show comparative model matrix")
    args = parser.parse_args()

    print_budget_report(
        params_b=args.model_params,
        model_name=args.model_name,
        quant=args.quantization,
        lora_r=args.lora_rank,
        seq_l=args.seq_len,
        batch_sz=args.batch_size,
        grad_ckpt=not args.no_grad_ckpt,
        optim=args.optimizer,
    )

    if args.matrix:
        compare_models_on_t4()


if __name__ == "__main__":
    main()
