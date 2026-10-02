# Dataset Documentation: Russian Customer Support Ticket Preference Corpus

## 1. Provenance & Description
* **Task Type**: Direct Preference Optimization (DPO) pairwise preference alignment.
* **Domain**: E-commerce, Retail & Helpdesk Customer Support in Russian.
* **Language**: Russian (`ru`).
* **Format**: Standard JSON Lines (`.jsonl`) with schema `{"prompt": str, "chosen": str, "rejected": str}`.

---

## 2. File Manifest & Checksums (SHA-256)

| File | Number of Pairs | SHA-256 Checksum | Purpose |
| :--- | :--- | :--- | :--- |
| `train.jsonl` | 450 pairs | `862ab67a67cdf184d987b591fd1d46820d559df3a5e5d0f2928ad1b5bad65a86` | DPO fine-tuning training set |
| `eval.jsonl` | 50 pairs | `1ff4942e4a9db598aa1f5652d8797e4ac118d8a2246d7a15d568756540bed296` | Deterministic held-out evaluation set |

---

## 3. Ticket Categories & Coverage

1. **Order Status & Delayed Delivery Tracking** (e.g. *"Где мой заказ #...?"*)
2. **Refunds & Billing Issues** (e.g. *"Когда вернутся деньги на карту?"*)
3. **Courier Delivery Rescheduling** (e.g. *"Курьер не приехал в назначенное время"*)
4. **App & Website Technical Errors** (e.g. *"Ошибка Error 503 при оплате"*)
5. **Damaged / Defective Items & Warranty Claims** (e.g. *"Экран разбит, коробка помята"*)
6. **Returns & 14-Day Exchange Policy** (e.g. *"Одежда не подошла по размеру"*)
7. **2FA Security Alerts & Unauthorized Access** (e.g. *"Уведомление о входе из другого города"*)
8. **Lost Credentials & SMS Verification Issues** (e.g. *"Не приходит SMS с кодом"*)

---

## 4. Summary Statistics & Length Distribution

```text
================================================================================
 DATASET SUMMARY STATISTICS (500 Total Validated Pairs)
================================================================================
Total Valid Rows:             500 (0 malformed, 0 schema errors)
Identical Chosen/Rejected:    0 (100.0% distinct pairs)
Average Prompt Length:        17.5 tokens (93.4 characters)
Average Chosen Length:        42.9 tokens (241.6 characters)
Average Rejected Length:      17.4 tokens (98.2 characters)
Max Total Sequence Length:    183 tokens (< 1024 max_length bound)
Truncation Rate (at 1024):    0.0% (Zero data loss)
================================================================================
```

---

## 5. Alignment Philosophy

* **Chosen Responses**: Empathetic, polite greeting (*"Здравствуйте!"*), explicit acknowledgment of the user's issue, concrete multi-step actionable guidance, specific compensation/bonus points where applicable, and professional closure.
* **Rejected Responses**: Terse, dismissive, shifting blame to the user (*"Сами виноваты"*, *"Мы не несем ответственности"*), cold refusal without alternative pathways, or evasive non-answers (*"Ждите"*).
