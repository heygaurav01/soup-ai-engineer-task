#!/usr/bin/env python3
"""inspect_dataset.py

Comprehensive dataset inspection and validation utility for DPO preference datasets.
Analyzes token lengths, length bias, schema integrity, and lexical overlap.
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DatasetInspector")


def simple_tokenize(text: str) -> List[str]:
    """Fallback tokenizer if transformers is unavailable or offline."""
    import re
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def get_tokenizer(model_id: Optional[str] = None):
    """Load HF tokenizer if available, else return fallback function."""
    if model_id:
        try:
            from transformers import AutoTokenizer
            logger.info(f"Loading tokenizer from '{model_id}'...")
            return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Could not load HuggingFace tokenizer '{model_id}': {e}. Using fallback tokenizer.")
    return None


def calculate_stats(lengths: List[int]) -> Dict[str, float]:
    """Calculate descriptive statistics for a list of integers."""
    if not lengths:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0}
    sorted_l = sorted(lengths)
    n = len(sorted_l)
    mean_val = sum(sorted_l) / n
    median_val = sorted_l[n // 2] if n % 2 != 0 else (sorted_l[n // 2 - 1] + sorted_l[n // 2]) / 2.0
    p90_idx = min(n - 1, int(math.ceil(0.90 * n)) - 1)
    p95_idx = min(n - 1, int(math.ceil(0.95 * n)) - 1)
    return {
        "count": n,
        "min": float(sorted_l[0]),
        "max": float(sorted_l[-1]),
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "p90": float(sorted_l[p90_idx]),
        "p95": float(sorted_l[p95_idx]),
    }


def analyze_file(file_path: Path, tokenizer=None) -> Dict[str, Any]:
    """Analyze a single JSONL preference dataset file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    records = []
    invalid_rows = 0
    missing_keys = 0
    duplicate_prompts = 0
    seen_prompts = set()
    identical_pairs = 0

    prompt_char_lens = []
    chosen_char_lens = []
    rejected_char_lens = []

    prompt_tok_lens = []
    chosen_tok_lens = []
    rejected_tok_lens = []

    chosen_longer_count = 0
    rejected_longer_count = 0
    equal_len_count = 0

    all_chosen_tokens = []
    all_rejected_tokens = []
    jaccard_overlaps = []

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                logger.error(f"Row {idx}: JSON parse error: {err}")
                invalid_rows += 1
                continue

            if not all(k in row for k in ("prompt", "chosen", "rejected")):
                logger.error(f"Row {idx}: Missing required keys (prompt, chosen, rejected)")
                missing_keys += 1
                continue

            prompt = str(row.get("prompt", ""))
            chosen = str(row.get("chosen", ""))
            rejected = str(row.get("rejected", ""))

            if prompt in seen_prompts:
                duplicate_prompts += 1
            seen_prompts.add(prompt)

            if chosen == rejected:
                identical_pairs += 1

            prompt_char_lens.append(len(prompt))
            chosen_char_lens.append(len(chosen))
            rejected_char_lens.append(len(rejected))

            if tokenizer:
                p_toks = tokenizer.encode(prompt, add_special_tokens=False)
                c_toks = tokenizer.encode(chosen, add_special_tokens=False)
                r_toks = tokenizer.encode(rejected, add_special_tokens=False)
                c_tok_words = tokenizer.tokenize(chosen)
                r_tok_words = tokenizer.tokenize(rejected)
            else:
                p_toks = simple_tokenize(prompt)
                c_toks = simple_tokenize(chosen)
                r_toks = simple_tokenize(rejected)
                c_tok_words = c_toks
                r_tok_words = r_toks

            p_len, c_len, r_len = len(p_toks), len(c_toks), len(r_toks)
            prompt_tok_lens.append(p_len)
            chosen_tok_lens.append(c_len)
            rejected_tok_lens.append(r_len)

            if c_len > r_len:
                chosen_longer_count += 1
            elif r_len > c_len:
                rejected_longer_count += 1
            else:
                equal_len_count += 1

            # Lexical overlap
            set_c = set(c_tok_words)
            set_r = set(r_tok_words)
            union_len = len(set_c.union(set_r))
            jaccard = (len(set_c.intersection(set_r)) / union_len) if union_len > 0 else 0.0
            jaccard_overlaps.append(jaccard)

            all_chosen_tokens.extend(c_tok_words)
            all_rejected_tokens.extend(r_tok_words)

            records.append(row)

    total_valid = len(records)
    length_bias_pct = (chosen_longer_count / total_valid * 100) if total_valid > 0 else 0.0

    # Type-Token Ratio (Lexical Diversity)
    ttr_chosen = (len(set(all_chosen_tokens)) / len(all_chosen_tokens)) if all_chosen_tokens else 0.0
    ttr_rejected = (len(set(all_rejected_tokens)) / len(all_rejected_tokens)) if all_rejected_tokens else 0.0
    avg_jaccard = (sum(jaccard_overlaps) / len(jaccard_overlaps)) if jaccard_overlaps else 0.0

    return {
        "file_name": file_path.name,
        "total_samples": total_valid,
        "invalid_json_rows": invalid_rows,
        "missing_key_rows": missing_keys,
        "duplicate_prompts": duplicate_prompts,
        "identical_chosen_rejected": identical_pairs,
        "length_bias": {
            "chosen_longer": chosen_longer_count,
            "rejected_longer": rejected_longer_count,
            "equal_length": equal_len_count,
            "chosen_longer_pct": round(length_bias_pct, 2),
        },
        "lexical_metrics": {
            "chosen_type_token_ratio": round(ttr_chosen, 4),
            "rejected_type_token_ratio": round(ttr_rejected, 4),
            "mean_jaccard_similarity": round(avg_jaccard, 4),
        },
        "token_stats": {
            "prompt": calculate_stats(prompt_tok_lens),
            "chosen": calculate_stats(chosen_tok_lens),
            "rejected": calculate_stats(rejected_tok_lens),
            "total_pair_seq_length": calculate_stats([p + max(c, r) for p, c, r in zip(prompt_tok_lens, chosen_tok_lens, rejected_tok_lens)]) if prompt_tok_lens else {},
        },
        "char_stats": {
            "prompt": calculate_stats(prompt_char_lens),
            "chosen": calculate_stats(chosen_char_lens),
            "rejected": calculate_stats(rejected_char_lens),
        },
    }


def print_ascii_table(title: str, stats: Dict[str, Any]):
    """Print clean ASCII summary tables."""
    print("=" * 78)
    print(f" DATASET INSPECTION REPORT: {title.upper()}")
    print("=" * 78)
    print(f"Total Valid Pairs:       {stats['total_samples']}")
    print(f"Invalid Rows / Errors:   {stats['invalid_json_rows']}")
    print(f"Missing Schema Keys:     {stats['missing_key_rows']}")
    print(f"Duplicate Prompts:       {stats['duplicate_prompts']}")
    print(f"Identical Pairs:         {stats['identical_chosen_rejected']}")
    print("-" * 78)
    print("TOKEN LENGTH DISTRIBUTION:")
    print(f"{'Field':<12} | {'Min':<6} | {'Max':<6} | {'Mean':<8} | {'Median':<8} | {'P90':<6} | {'P95':<6}")
    print("-" * 78)
    for field in ["prompt", "chosen", "rejected"]:
        ts = stats["token_stats"][field]
        print(f"{field.capitalize():<12} | {ts['min']:<6.0f} | {ts['max']:<6.0f} | {ts['mean']:<8.1f} | {ts['median']:<8.1f} | {ts['p90']:<6.0f} | {ts['p95']:<6.0f}")
    if "total_pair_seq_length" in stats["token_stats"] and stats["token_stats"]["total_pair_seq_length"]:
        ts = stats["token_stats"]["total_pair_seq_length"]
        print(f"{'Max Pair Seq':<12} | {ts['min']:<6.0f} | {ts['max']:<6.0f} | {ts['mean']:<8.1f} | {ts['median']:<8.1f} | {ts['p90']:<6.0f} | {ts['p95']:<6.0f}")
    print("-" * 78)
    lb = stats["length_bias"]
    print(f"Length Bias Check: Chosen longer: {lb['chosen_longer']} ({lb['chosen_longer_pct']}%), "
          f"Rejected longer: {lb['rejected_longer']}, Equal: {lb['equal_length']}")
    lex = stats["lexical_metrics"]
    print(f"Lexical Metrics:   Chosen TTR: {lex['chosen_type_token_ratio']}, "
          f"Rejected TTR: {lex['rejected_type_token_ratio']}, Mean Jaccard: {lex['mean_jaccard_similarity']}")
    print("=" * 78)
    print()


def main():
    parser = argparse.ArgumentParser(description="Inspect DPO preference datasets.")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).parent.parent / "data"),
                        help="Path to data directory containing train.jsonl and eval.jsonl")
    parser.add_argument("--train-file", type=str, default="train.jsonl", help="Train dataset filename")
    parser.add_argument("--eval-file", type=str, default="eval.jsonl", help="Eval dataset filename")
    parser.add_argument("--tokenizer", type=str, default=None, help="HuggingFace model ID or tokenizer path")
    parser.add_argument("--output-json", type=str, default=None, help="Path to write JSON inspection report")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / args.train_file
    eval_path = data_dir / args.eval_file

    tokenizer = get_tokenizer(args.tokenizer)

    results = {}
    if train_path.exists():
        logger.info(f"Inspecting training dataset: {train_path}")
        results["train"] = analyze_file(train_path, tokenizer=tokenizer)
        print_ascii_table("Training Set (train.jsonl)", results["train"])
    else:
        logger.warning(f"Train file not found at: {train_path}")

    if eval_path.exists():
        logger.info(f"Inspecting evaluation dataset: {eval_path}")
        results["eval"] = analyze_file(eval_path, tokenizer=tokenizer)
        print_ascii_table("Evaluation Set (eval.jsonl)", results["eval"])
    else:
        logger.warning(f"Eval file not found at: {eval_path}")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved inspection summary JSON to: {out_p}")


if __name__ == "__main__":
    main()
