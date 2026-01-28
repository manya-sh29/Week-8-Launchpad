# DAY 2 —  QLoRA Fine-Tuning 

## Overview
This repository demonstrates parameter-efficient fine-tuning of a LLaMA-based model using QLoRA and PEFT. The goal is to adapt large language models on limited GPU memory by updating only a small subset of parameters (LoRA adapters) while using 4‑bit quantization.

- BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


## Features
- QLoRA with 4-bit (nf4) quantization
- LoRA adapters (low-rank updates) for parameter-efficient training
- Mixed-precision (FP16) training
- Memory optimizations for limited-GPU environments
- Example notebook and saved adapter weights

## Repository structure
- /notebooks/lora_train.ipynb — example training notebook
- /adapters/adapter_model.bin — saved LoRA adapter weights
- /TRAINING-REPORT.md — original training notes and report

## Requirements
- Python 3.8+
- A CUDA-enabled GPU (T4 or better recommended)
- Core packages:
    - torch
    - transformers
    - accelerate
    - peft
    - bitsandbytes

Install with:

pip install -r requirements.txt

(or)

pip install torch transformers accelerate peft bitsandbytes


## Quickstart
1. Configure Accelerate (choose your device and 1 process per GPU):
```
accelerate config
```
2. Run the training notebook or script:
- Open `/notebooks/lora_train.ipynb` in Colab or Jupyter


## Recommended Configuration
- Base model: TinyLlama / LLaMA-based model (4-bit nf4 quantized)
- Compute dtype: float16
- Mixed precision enabled


## Memory optimization tips
- Load model in 4-bit using bitsandbytes
- Enable gradient checkpointing
- Freeze base model and train only LoRA adapters
- Use mixed precision (FP16)
- Monitor GPU utilization and reduce batch size if needed

## Outputs
- Notebook: /notebooks/lora_train.ipynb
- Adapter weights: /adapters/adapter_model.bin
- Training report: /TRAINING-REPORT.md


