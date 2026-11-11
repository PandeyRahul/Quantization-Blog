# INT8, INT4, GGUF or ONNX? — code for the blog

Code for my Medium post on LLM quantization formats.

| File | What it does |
|---|---|
| `quant_error.py` | Quantizes one 1024×4096 weight matrix (with 0.1% outliers) 8+ ways — naive INT8/INT4, per-channel, GGUF Q8_0/Q5_x/Q4_x — and measures the error. Saves `quant_error_results.json`. |
| `make_images.py` | Builds all blog figures into `img/` (needs `quant_error_results.json`). |
| `bench_ollama.py` | Real-model benchmark: Qwen2.5 1.5B at FP16 / Q8_0 / Q4_K_M — file size, VRAM, tokens/sec, 5-question quiz. |

## Run

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows  (Linux/mac: source .venv/bin/activate)
pip install -r requirements.txt

python quant_error.py          # weight-error experiment
python make_images.py          # charts -> img/

# real model benchmark (needs Ollama running + an NVIDIA GPU for the VRAM column)
ollama pull qwen2.5:1.5b-instruct-fp16
ollama pull qwen2.5:1.5b-instruct-q8_0
ollama pull qwen2.5:1.5b-instruct-q4_K_M
python bench_ollama.py
```

Check exact tag names at https://ollama.com/library/qwen2.5/tags if a pull fails.

## ONNX snippet from the post

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("model.onnx", "model_int8.onnx",
                 weight_type=QuantType.QInt8, per_channel=True)
```
