"""Benchmark the SAME model at different quantization levels with Ollama.

Run on your own machine:
    pip install ollama
    ollama pull qwen2.5:1.5b-instruct-fp16
    ollama pull qwen2.5:1.5b-instruct-q8_0
    ollama pull qwen2.5:1.5b-instruct-q4_K_M
    python bench_ollama.py

(Check exact tag names on https://ollama.com/library/qwen2.5/tags)
"""
import subprocess
import ollama

MODELS = [
    "qwen2.5:1.5b-instruct-fp16",
    "qwen2.5:1.5b-instruct-q8_0",
    "qwen2.5:1.5b-instruct-q4_K_M",
]

# Small fixed quiz so we can eyeball quality, not just speed
QUIZ = [
    ("What is 17 * 23? Reply with only the number.", "391"),
    ("What is the capital of Australia? One word.", "canberra"),
    ("Is 97 a prime number? Reply yes or no.", "yes"),
    ("Spell 'quantization' backwards. Only the word.", "noitazitnauq"),
    ("Which planet is known as the Red Planet? One word.", "mars"),
]
PROMPT = "Explain quantization of neural networks to a 10 year old in about 150 words."
RUNS = 3


def vram_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


for model in MODELS:
    # warm up (loads the model into memory)
    ollama.generate(model=model, prompt="hi", options={"num_predict": 1})
    vram = vram_used_mb()

    speeds = []
    for _ in range(RUNS):
        r = ollama.generate(model=model, prompt=PROMPT,
                            options={"temperature": 0, "num_predict": 200, "seed": 42})
        speeds.append(r["eval_count"] / (r["eval_duration"] / 1e9))  # tokens/sec

    correct = 0
    for q, ans in QUIZ:
        r = ollama.generate(model=model, prompt=q, options={"temperature": 0, "num_predict": 20})
        correct += ans in r["response"].lower()

    size_gb = next((m.size for m in ollama.list().models if m.model == model), 0) / 1e9
    print(f"{model:32s} | file {size_gb:5.2f} GB | VRAM {vram} MB | "
          f"{sum(speeds)/len(speeds):6.1f} tok/s | quiz {correct}/{len(QUIZ)}")

    ollama.generate(model=model, prompt="", keep_alive=0)  # unload before the next one
