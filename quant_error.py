"""How much does each quantization scheme damage a weight matrix?
Same matrix, different schemes -> relative reconstruction error (lower = better)."""
import json
import numpy as np
from gguf import quants, GGMLQuantizationType as T

rng = np.random.default_rng(42)

def make_weights(outliers=True):
    W = rng.normal(0, 0.02, size=(1024, 4096)).astype(np.float32)
    if outliers:  # real LLM weights have a few huge values -> simulate 0.1% outliers
        m = rng.random(W.shape) < 0.001
        W[m] *= 25
    return W

def rel_err(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))

def absmax(W, bits, axis=None):
    q = 2 ** (bits - 1) - 1
    s = np.abs(W).max(axis=axis, keepdims=axis is not None) / q
    return np.clip(np.round(W / s), -q, q) * s

def run(W):
    r = {"FP16": rel_err(W, W.astype(np.float16).astype(np.float32))}
    r["INT8 per-tensor"] = rel_err(W, absmax(W, 8))
    r["INT8 per-channel"] = rel_err(W, absmax(W, 8, axis=1))
    r["INT4 per-tensor"] = rel_err(W, absmax(W, 4))
    r["INT4 per-channel"] = rel_err(W, absmax(W, 4, axis=1))
    for name, t in [("Q8_0", T.Q8_0), ("Q5_1", T.Q5_1), ("Q5_0", T.Q5_0), ("Q4_1", T.Q4_1), ("Q4_0", T.Q4_0)]:
        qd = quants.quantize(W, t)
        r[f"GGUF {name}"] = rel_err(W, quants.dequantize(qd, t).reshape(W.shape))
        r[f"GGUF {name} (bits/weight)"] = qd.nbytes * 8 / W.size
    return r

out = {"with_outliers": run(make_weights(True)), "no_outliers": run(make_weights(False))}
for k in out["with_outliers"]:
    print(f"{k:28s} outliers={out['with_outliers'][k]:.4f}   clean={out['no_outliers'][k]:.4f}")
json.dump(out, open("quant_error_results.json", "w"), indent=2)
