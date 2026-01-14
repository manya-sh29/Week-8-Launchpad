import json
import matplotlib.pyplot as plt

DATA_FILE = "data/train.jsonl"

token_lengths = []

# Read dataset
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        text = item["instruction"] + " " + item["input"] + " " + item["output"]
        token_lengths.append(len(text.split()))

# -------- Histogram --------
plt.figure()
plt.hist(token_lengths, bins=30)
plt.xlabel("Token Count")
plt.ylabel("Number of Samples")
plt.title("Token Length Distribution")
plt.savefig("data/token_length_distribution.png")
plt.close()

print("Token length distribution graph saved.")
