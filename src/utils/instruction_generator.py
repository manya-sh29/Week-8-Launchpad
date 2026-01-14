import json
import random

RAW_FILE = "/home/manyasharma/Desktop/Week8-tasks/src/data/qa_cleaned.json"
TRAIN_FILE = "/home/manyasharma/Desktop/Week8-tasks/src/data/train.jsonl"
VAL_FILE = "/home/manyasharma/Desktop/Week8-tasks/src/data/val.jsonl"

MAX_SAMPLES = 1000
TRAIN_RATIO = 0.9 
samples = []

with open(RAW_FILE, "r", encoding="utf-8") as rf:
    for line in rf:
        if len(samples) >= MAX_SAMPLES:
            break

        item = json.loads(line)

        samples.append({
            "instruction": "Answer the following programming question.",
            "input": item["question"].strip(),
            "output": item["answer"].strip()
        })

random.shuffle(samples)
split_idx = int(len(samples) * TRAIN_RATIO)

train_samples = samples[:split_idx]
val_samples = samples[split_idx:]

with open(TRAIN_FILE, "w", encoding="utf-8") as tf:
    for s in train_samples:
        tf.write(json.dumps(s, ensure_ascii=False) + "\n")

with open(VAL_FILE, "w", encoding="utf-8") as vf:
    for s in val_samples:
        vf.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Train samples: {len(train_samples)}")
print(f"Validation samples: {len(val_samples)}")
