# DAY 3 — QUANTISATION (8-bit → 4-bit → GGUF)

## Overview
This report documents the quantisation of a Large Language Model (LLM) from higher precision (FP16) to lower-precision formats (INT8, INT4, and GGUF).  
The goal is to reduce memory usage and improve inference speed while maintaining acceptable output quality.

---

##  Quantisation
1. Load FP16 base model
2. Choose quantisation method (static / dynamic)
3. Apply post-training quantisation
4. Convert model to INT8
5. Save INT8 model
6. Convert model to INT4
7. Save INT4 model
8. Convert model to GGUF (q4_0 / q8_0)
9. Save GGUF model


---

## Topics Covered
- Post-training quantisation
- Static vs dynamic quantisation
- FP16 vs INT8 vs INT4 comparison
- GGUF format and llama.cpp conversion workflow

---


## Precision Formats
- **FP16**: High accuracy, high memory usage
- **INT8**: Balanced accuracy and memory
- **INT4**: Maximum compression, slight quality drop
- **GGUF**: Optimized format for llama.cpp (CPU inference)

---


## QUANTISATION Workflow
Quantisation Workflow

- Step 1: FP16 Model Ready
Start with the base model in FP16 format (baseline for all conversions).

- Step 2: Choose Quantisation Method
Select static (calibration-based) or dynamic (runtime) quantisation.

- Step 3: Post-Training Quantisation
Apply the chosen method to reduce model precision and memory usage.

- Step 4: Convert to INT8
Quantise FP16 weights to 8-bit and save output.
 /quantized/model-int8/

- Step 5: Convert to INT4
Quantise weights to 4-bit for further size reduction.
 /quantized/model-int4/

- Step 6: Convert to GGUF
Convert quantised model to GGUF format (q4_0 or q8_0) for llama.cpp support.
 /quantized/model.gguf/

- Step 7: Evaluate Models
Compare FP16, INT8, INT4, and GGUF models based on size, speed, and quality.

- Step 8: Document Results
Record observations and comparisons.
/quantized/QUANTISATION-REPORT.md

## Exercise Performed

The base model was converted into the following formats:

- INT8 quantised model
- INT4 quantised model
- GGUF model (q4_0 / q8_0)

Each version was evaluated on:
- Model size
- Inference speed
- Output quality

---
``` 
Deliverables
/quantized/model-int8
/quantized/model-int4
/quantized/model.gguf
QUANTISATION-REPORT.md

```
