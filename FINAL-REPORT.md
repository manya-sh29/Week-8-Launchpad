# Day 5 — Capstone: Build & Deploy Local LLM API

## Overview
Build and deploy a **local LLM API** as a microservice supporting **text generation** and **chat**, optimized for fast inference and ready for **RAG & agents**.

## Learning Outcomes
- Deploy LLM as a microservice using FastAPI/Flask  
- Streamed generation & infinite chat  
- Model caching & quantization  
- Practical implementation of system + user prompts  


## API Endpoints

- **POST /generate** –
```
 -Accepts a POST request with a system prompt and user prompt.
 -Generates a unique request ID for tracking.
 -Builds the messages in the format required by the LLM.
 -Logs the request for debugging or monitoring purposes.
 -Returns the model’s response as a streaming text output, allowing users to see the text as it’s generated in real time.
 ```

- **POST /chat** – 
```
 -Handles conversation with the model using system and user prompts.  
 -Maintains chat history to support continuous/infinite chat sessions.  
 -Streams the assistant’s response in real time and logs request details.
 ```



## Streamlit UI
- Implemented for interactive testing of /generate and /chat endpoints
- Allows real-time testing with streaming output

## Working Screenshot
![alt text](<Screenshot from 2026-01-28 12-37-51.png>)

## Features
- Quantized model for faster inference  
- Top-k, Top-p, Temperature controls  
- Logs with request ID  
- Ready for RAG & agent integration  

Logging

- All requests are logged with timestamps and unique request IDs in:
- src/logs/llm.logs




