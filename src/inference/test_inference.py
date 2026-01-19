import json
import asyncio
import time
import csv
import os
from openai import AsyncOpenAI
from typing import List, Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


RESULT_FILE = "src/benchmarks/results.csv"
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def semantic_accuracy(generated_answer: str, expected_answer: str) -> float:
    if not expected_answer:
        return None

    embeddings = EMBEDDING_MODEL.encode(
        [generated_answer, expected_answer],
        normalize_embeddings=True,
    )

    gen_vec, exp_vec = embeddings
    dim = gen_vec.shape[0]

    index = faiss.IndexFlatIP(dim)
    index.add(np.array([exp_vec]))

    similarity, _ = index.search(np.array([gen_vec]), k=1)
    return float(similarity[0][0])


def chunk_list(data: List[Dict], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]


def append_to_csv(row: dict):
    file_exists = os.path.exists(RESULT_FILE)
    with open(RESULT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


async def stream_one_prompt(
    LLM: AsyncOpenAI,
    prompt: str,
    expected: str,
    prompt_id: int,
    batch_id: int,
):
    buffer = []
    token_count = 0
    start_time = time.time()
    first_token_time = None

    stream = await LLM.chat.completions.create(
        model="src/quantized/model-q4_0.gguf",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.7,
        stream=True,
    )

    async for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        if delta and delta.content:
            if first_token_time is None:
                first_token_time = time.time()

            buffer.append(delta.content)
            token_count += len(delta.content.split())

            print(
                f"[Batch {batch_id} | Prompt {prompt_id}] {delta.content}",
                end="",
                flush=True,
            )

    end_time = time.time()
    latency = end_time - start_time
    ttft = first_token_time - start_time if first_token_time else None
    tokens_per_sec = token_count / latency if latency > 0 else 0

    generated_answer = "".join(buffer)
    accuracy = semantic_accuracy(generated_answer, expected)

    print("\n" + "-" * 80)

    append_to_csv(
        {
            "model": "model-q4_0.gguf",
            "prompt_id": prompt_id,
            "prompt": prompt,
            "tokens": token_count,
            "latency_sec": round(latency, 3),
            "ttft_sec": round(ttft, 3) if ttft else None,
            "tokens_per_sec": round(tokens_per_sec, 2),
            "semantic_accuracy": round(accuracy, 4) if accuracy is not None else None,
        }
    )


async def run_batch(
    LLM: AsyncOpenAI,
    prompts: List[Dict],
    batch_id: int,
):
    print(
        f"\nBatch {batch_id} ===============================================================\n"
    )

    tasks = [
        stream_one_prompt(
            LLM,
            p["prompt"],
            p.get("output", ""),
            i,
            batch_id,
        )
        for i, p in enumerate(prompts, start=1)
    ]

    await asyncio.gather(*tasks)


async def main():
    LLM = AsyncOpenAI(
        base_url="http://localhost:8080/v1",
        api_key="ms",
    )

    prompts = [
        {
            "prompt": "What is LLM hallucination?",
            "output": "LLM hallucination is when a language model generates incorrect or fabricated information."
        },
        {
            "prompt": "What is ChatGPT?",
            "output": "ChatGPT is a conversational AI model developed by OpenAI."
        },
        {
            "prompt": "What is Artificial Intelligence?",
            "output": "Artificial Intelligence is the simulation of human intelligence in machines."
        },
    ]

    BATCH_SIZE = 2

    for batch_id, prompt_batch in enumerate(
        chunk_list(prompts, BATCH_SIZE), start=1
    ):
        await run_batch(LLM, prompt_batch, batch_id)

    print("\nInference complete. Results appended to results.csv")


if __name__ == "__main__":
    asyncio.run(main())





# import asyncio
# from openai import AsyncOpenAI

# async def give_response(llm: AsyncOpenAI, prompt: str):
#     print(f"\nPrompt: {prompt}")
#     print("Response: ", end="", flush=True)
#     response = await llm.chat.completions.create(
#         model="src/quantized/model-int4",
#         messages=[{"role" : "user", "content" : "what is llm hallucination?"}],
#         max_tokens=256,
#         temperature=0.7,
#         stream=True,
#     )

#     async for chunk in response:
#         if not chunk.choices:
#             continue

#         delta = chunk.choices[0].delta
#         if delta and delta.content:
#             print(delta.content, end="", flush=True)

# async def main():
#     llm = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key="apikey")
#     prompt = ["What is LLM Halucination","What is chatgpt", "What is AI"]
#     for p in prompt:
#         await give_response(llm,p)

# if __name__ == "__main__":
#     asyncio.run(main())




