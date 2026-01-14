## Day 1 — LLM Architecture & Data Preparation for Fine-Tuning

### Objective
- Build an instruction-tuning dataset for fine-tuning LLMs using three instruction types: QA, Reasoning, and Extraction.
- Requirements:
    - Clean and curated
    - Domain-based: Coding
    - At least 1,000 samples
    - Prepared for tokenization, outlier removal, and distribution analysis

---

## 1. Dataset overview

| Attribute         | Description                                              |
|-------------------|----------------------------------------------------------|
| Domain            | Coding                                                   |
| Total raw samples | 1,000                                                    |
| Instruction types | QA, Reasoning, Extraction                                |
| Format            | JSONL (one JSON object per line: {"instruction":"...", "input":"...", "output":"."}) |
| Train file        | src/data/train.jsonl                                      |
| Validation file   | src/data/val.jsonl                                        |
| Helper script     | src/utils/data_cleaner.py                                 |

---

## 2. Instruction types and examples

Each dataset entry must be a JSON object with the keys: instruction, input, output.

1) QA
- Purpose: Standard question-answer pairs for direct responses.
- Example:
```json
{
    "instruction": "Answer the following programming question.",
    "input": "Is it possible to do a Call-with-current-continuation in Go?",
    "output": "No. Go does not provide first-class continuations, so call/cc is not supported."
}
```

2) Reasoning
- Purpose: Explain why an answer is correct — chain-of-thought style should be concise and focused on the final rationale (avoid exposing verbose internal chain-of-thought if planning to hide).
- Example:
```json
{
    "instruction": "Explain why the following answer is correct.",
    "input": "Is it possible to do a Call-with-current-continuation in Go?",
    "output": "Go's language design does not include first-class continuations or a runtime that exposes stack continuations, so typical call/cc semantics are not achievable."
}
```

3) Extraction
- Purpose: Extract key facts, conclusions, or concise summaries from an answer or text.
- Example:
```json
{
    "instruction": "Extract the final conclusion from the answer.",
    "input": "According to one of the Go contributors, no it's not possible.",
    "output": "Call-with-current-continuation is not possible in Go."
}
```

---

## 3. Preparation notes
- Tokenization: run tokenizer on sample subset to estimate token lengths and memory footprint.
- Outlier removal: identify and remove extremely long or malformed examples.
- Distribution analysis: ensure balanced representation across instruction types and common coding topics.
- Scripts:
    - Data cleaning and validation: src/utils/data_cleaner.py
    - Additional helpers: src/utils/instruction_generator.py

--- 

Keep JSONL entries compact and consistent; validate schema before splitting into train/validation.

```
project-root/
└── src/
    ├── data/
    │   ├── qa.en.go.json
    │   ├── train.jsonl
    │   └── val.jsonl
    │
    ├── utils/
    │   ├── instruction_generator.py
    │   └── data_cleaner.py
    │
    └── DATASET-ANALYSIS.md

```


![alt text](<Screenshot from 2026-01-10 23-47-12.png>)

---
![alt text](<Screenshot from 2026-01-10 23-43-59-1.png>)
