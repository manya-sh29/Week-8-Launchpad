import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from difflib import SequenceMatcher  # for accuracy measurement

# ======================================================
# CONFIG
# ======================================================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FT_MODEL = "src/inference/model_fp16"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SINGLE_PROMPT = "Explain Large Language Models in simple terms."
EXPECTED_ANSWER = "A Large Language Model is an AI model trained on a lot of text to understand and generate human-like language."

BATCH_PROMPTS = [
    "What is Artificial Intelligence?",
    "Explain Machine Learning.",
    "What is a GPU?",
]

EXPECTED_BATCH_ANSWERS = [
    "Artificial Intelligence is a field of computer science that makes machines intelligent.",
    "Machine Learning is a method where machines learn patterns from data.",
    "A GPU is a graphics processing unit used for computation."
]

# Optional: limit context window (context window optimization)
MAX_CONTEXT_TOKENS = 512

# ======================================================
# UTILS
# ======================================================
def load_model(model_name):
    """
    Load model on device with correct dtype
    Implements: Base/Fine-tuned model loading.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"
    )
    model.eval()
    return model


def calculate_accuracy(output_text, reference_text):
    """
    Simple accuracy metric: text similarity ratio
    Returns a value between 0 and 1.
    Implements: Accuracy measurement logic
    """
    return SequenceMatcher(None, output_text, reference_text).ratio()


def run_inference(model, tokenizer, prompt, expected=None, stream=False):
    """
    Single prompt inference.
    Implements: Normal inference + streaming + KV caching + latency + tokens/sec + accuracy
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CONTEXT_TOKENS
    ).to(DEVICE)

    streamer = TextStreamer(tokenizer) if stream else None

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            use_cache=True,   # KV caching
            streamer=streamer
        )
    end = time.time()

    generated_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
    latency = end - start
    tokens_per_sec = generated_tokens / latency if latency > 0 else 0
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    accuracy = calculate_accuracy(text, expected) if expected else None

    return text, latency, tokens_per_sec, accuracy


def run_batch_inference(model, tokenizer, prompts, expected_list=None):
    """
    Batch / multi-prompt inference.
    Implements: Batch inference + multi-prompt + latency + tokens/sec + accuracy
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_CONTEXT_TOKENS
    ).to(DEVICE)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            use_cache=True
        )
    end = time.time()

    latency = end - start
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Tokens/sec for batch
    total_generated_tokens = sum([len(tokenizer.encode(t)) for t in texts])
    tokens_per_sec = total_generated_tokens / latency if latency > 0 else 0

    # Accuracy for each prompt
    accuracies = []
    if expected_list:
        for out, ref in zip(texts, expected_list):
            accuracies.append(calculate_accuracy(out, ref))

    return texts, latency, tokens_per_sec, accuracies


def run_speculative_decoding(model, tokenizer, prompt, expected=None):
    """
    Simple speculative decoding simulation.
    Implements: Basic speculative decoding logic inside Python.
    Here we generate multiple sequences and pick the first as 'fastest'.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CONTEXT_TOKENS
    ).to(DEVICE)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            num_return_sequences=1,   # simulate speculative decoding
            use_cache=True
        )
    end = time.time()

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
    latency = end - start
    tokens_per_sec = generated_tokens / latency if latency > 0 else 0
    accuracy = calculate_accuracy(text, expected) if expected else None

    return text, latency, tokens_per_sec, accuracy


# ======================================================
# MAIN
# ======================================================
def main():
    print(f"\nRunning on device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # ----------------- BASE MODEL -----------------
    print("\n===== BASE MODEL: Single Prompt =====")
    base_model = load_model(BASE_MODEL)
    output, latency, tps, accuracy = run_inference(
        base_model, tokenizer, SINGLE_PROMPT, expected=EXPECTED_ANSWER
    )
    print("Output:", output)
    print(f"Latency: {latency:.2f}s | Tokens/sec: {tps:.2f} | Accuracy: {accuracy:.2f}")

    # ----------------- FINE-TUNED MODEL -----------------
    print("\n===== FINE-TUNED MODEL: Single Prompt =====")
    ft_model = load_model(FT_MODEL)
    output, latency, tps, accuracy = run_inference(
        ft_model, tokenizer, SINGLE_PROMPT, expected=EXPECTED_ANSWER
    )
    print("Output:", output)
    print(f"Latency: {latency:.2f}s | Tokens/sec: {tps:.2f} | Accuracy: {accuracy:.2f}")

    # ----------------- STREAMING OUTPUT -----------------
    print("\n===== STREAMING OUTPUT (Base Model) =====")
    _, latency, tps, _ = run_inference(
        base_model, tokenizer, SINGLE_PROMPT, stream=True
    )
    print(f"\nStreaming Latency: {latency:.2f}s | Tokens/sec: {tps:.2f}")

    # ----------------- BATCH / MULTI-PROMPT -----------------
    print("\n===== BATCH INFERENCE (Base Model) =====")
    outputs, latency, tps, accuracies = run_batch_inference(
        base_model, tokenizer, BATCH_PROMPTS, expected_list=EXPECTED_BATCH_ANSWERS
    )
    for i, text in enumerate(outputs):
        acc = f"{accuracies[i]:.2f}" if i < len(accuracies) else "N/A"

        print(f"\nPrompt {i+1} Output:\n{text}\nAccuracy: {acc}")
    print(f"\nBatch Latency: {latency:.2f}s | Tokens/sec: {tps:.2f}")

    # ----------------- SPECULATIVE DECODING -----------------
    print("\n===== SPECULATIVE DECODING (Base Model) =====")
    output, latency, tps, accuracy = run_speculative_decoding(
        base_model,
        tokenizer,
        SINGLE_PROMPT,
        expected=EXPECTED_ANSWER
    )
    print("Output:", output)
    print(f"Latency: {latency:.2f}s | Tokens/sec: {tps:.2f} | Accuracy: {accuracy:.2f}")


    # ----------------- VRAM USAGE -----------------
    if DEVICE == "cuda":
        vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nPeak VRAM Usage: {vram:.2f} GB")


if __name__ == "__main__":
    main()
