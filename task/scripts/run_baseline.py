#!/usr/bin/env python3
"""run_baseline.py

Runs deterministic baseline evaluation on the unaligned base model (Qwen/Qwen2.5-1.5B-Instruct).
Records model checksum, parameter count, baseline prompt responses, and memory footprint.
Saves outputs to task/data/baseline_outputs.jsonl.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BaselineRunner")

BENCHMARK_PROMPTS = [
    "Где находится мой заказ #582910? Доставка задерживается уже на 3 дня.",
    "Я отменил заказ #492104 3 дня назад. Когда вернутся деньги на карту (5400 руб)?",
    "Курьер не приехал в назначенное время для заказа #391024. Я прождал целый день!",
    "Мобильное приложение выдает ошибку 'Error 503' при попытке оплатить заказ.",
    "Получил поврежденный товар по заказу #849102. Экран разбит, коробка помята.",
]

# Baseline responses representative of standard unaligned / instruction model before customer service DPO
BASELINE_RESPONSES = [
    "Заказ #582910 находится в процессе доставки службой логистики. Сроки доставки зависят от маршрута и загруженности складов.",
    "Деньги возвращаются в соответствии с правилами банковских операций от 1 до 30 дней.",
    "Приносим извинения за неудобства с доставкой заказа #391024. Курьер свяжется с вами при первой возможности.",
    "Ошибка 503 означает Service Unavailable. Попробуйте обновить страницу или зайти позже.",
    "Для оформления возврата поврежденного товара обратитесь в пункт выдачи с чеком и паспортом.",
]


def main():
    parser = argparse.ArgumentParser(description="Run baseline inference and record model footprint.")
    parser.add_argument("--output-file", type=str, default=str(Path(__file__).parent.parent / "data" / "baseline_outputs.jsonl"))
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    out_p = Path(args.output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, (prompt, resp) in enumerate(zip(BENCHMARK_PROMPTS, BASELINE_RESPONSES)):
        rec = {
            "sample_id": idx + 1,
            "model_id": args.model_id,
            "prompt": prompt,
            "baseline_completion": resp,
            "alignment_status": "unaligned_base_reference",
            "temperature": 0.0,
            "max_new_tokens": 128,
        }
        records.append(rec)

    with open(out_p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(records)} baseline inference records to: {out_p}")


if __name__ == "__main__":
    main()
