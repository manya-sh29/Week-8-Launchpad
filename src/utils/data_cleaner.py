import json
import os


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RAW_FILE = os.path.join(BASE_DIR, "src", "data", "qa.en.go.json")
CLEAN_FILE = os.path.join(BASE_DIR, "src", "data", "qa_cleaned.json")

MIN_LENGTH = 5 
cleaned_samples = []

if not os.path.exists(RAW_FILE):
    raise FileNotFoundError(f"Raw file not found at: {RAW_FILE}")

with open(RAW_FILE, "r", encoding="utf-8") as rf:
    for line in rf:
        try:
            item = json.loads(line)
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()

            if not question or not answer:
                continue
            if len(question) < MIN_LENGTH or len(answer) < MIN_LENGTH:
                continue

            cleaned_samples.append({
                "question": question,
                "answer": answer
            })

        except json.JSONDecodeError:
            continue

with open(CLEAN_FILE, "w", encoding="utf-8") as cf:
    for sample in cleaned_samples:
        cf.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"Cleaning completed: {len(cleaned_samples)} samples saved to {CLEAN_FILE}")
