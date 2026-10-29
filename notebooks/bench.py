import time

import torch

from src.verification.verifier import _get_nli

tokenizer, model, _ = _get_nli()
enc = tokenizer(["A short passage about attention mechanisms in neural networks."] * 6,
                ["Attention projects queries, keys and values."] * 6,
                return_tensors="pt", padding=True, truncation=True, max_length=128)

for threads in (1, 4, 8):
    torch.set_num_threads(threads)
    with torch.inference_mode():
        model(**enc)                       # warm-up
        t = time.perf_counter()
        model(**enc)
    print(f"threads={threads}: {time.perf_counter() - t:.2f}s")