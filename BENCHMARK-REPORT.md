# DAY 4 — BENCHMARK-REPORT.md

# AsyncOpenAI client
    - Instantiated with AsyncOpenAI(base_url="http://localhost:8080/v1", api_key="ms").
    - Calls the OpenAI-compatible endpoint to perform chat completions.

# Model reference
    - Uses a GGUF quantized model identifier when requesting completions: `model="src/quantized/model-q4_0.gguf"`.

# Streaming inference
    - Uses `stream=True` on `LLM.chat.completions.create(...)`.
    - Iterates `async for chunk in stream`, reads `chunk.choices[0].delta.content`, prints incremental tokens and accumulates them into a buffer.

# Token counting, latency and throughput
    - Counts tokens by splitting the incremental content on whitespace.
    - Measures:
        - `latency_sec` = total time from request start to stream end
        - `ttft_sec` = time to first token (first_token_time - start_time), if available
        - `tokens_per_sec` = token_count / latency_sec

# Semantic accuracy
    - Function `semantic_accuracy(generated_answer: str, expected_answer: str) -> float`
    - Embeddings obtained from `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")` with `normalize_embeddings=True`.
    - Similarity computed with FAISS `IndexFlatIP` over normalized embeddings (cosine similarity via inner product).
    - Returns a single float similarity score or `None` when `expected_answer` is empty.

# Batching and concurrency
    - `chunk_list(data, batch_size)` yields batches.
    - `run_batch(...)` launches concurrent streams for each prompt in a batch with `asyncio.gather`.
    - `BATCH_SIZE` is defined in `main()` (current code uses `BATCH_SIZE = 2`).

# CSV persistence (per prompt)
    - Results file: `RESULT_FILE = "src/benchmarks/results.csv"`.
    - `append_to_csv(row: dict)` opens the CSV in append mode, writes header if file did not exist, and appends one row per prompt with these fields:
        - `model`, `prompt_id`, `prompt`, `tokens`, `latency_sec`, `ttft_sec`, `tokens_per_sec`, `semantic_accuracy`

# Fixed generation params
    - `max_tokens=256`
    - `temperature=0.7`

# Entrypoint
    - Script calls `asyncio.run(main())` when executed as `__main__`.
    - `main()` prepares a small `prompts` list (3 examples) and iterates batches, calling `run_batch`.


# Metrics Measured
The following metrics were recorded for each model:

- **Latency (seconds)** – Total time to generate output
- **Tokens/sec** – Throughput of token generation
- **VRAM Usage (GB)** – Peak GPU memory usage
- **Accuracy** – Similarity between generated output and expected answer

# Results Saved
All benchmark results are appended to:
src/benchmarks/results.csv



