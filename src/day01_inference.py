# src/day01_inference.py
# Day 1: Understanding what a model actually IS

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilgpt2"

print("=" * 50)
print("Loading model and tokenizer...")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

print(f"\nModel loaded!")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Size (fp32): ~{sum(p.numel() * 4 for p in model.parameters()) / 1e6:.0f} MB")

# ---- STEP 1: Tokenization ----
text = "def fibonacci(n):"
tokens = tokenizer(text, return_tensors="pt")

print(f"\n--- TOKENIZATION ---")
print(f"Input text:  '{text}'")
print(f"input_ids:    {tokens['input_ids']}")
print(f"Token count:  {tokens['input_ids'].shape[1]}")

print(f"\nToken breakdown:")
for i, token_id in enumerate(tokens['input_ids'][0]):
    print(f"  Token {i}: id={token_id.item():5d} -> '{tokenizer.decode([token_id])}'")

# ---- STEP 2: Forward Pass ----
print(f"\n--- FORWARD PASS ---")
start = time.time()
with torch.no_grad():
    outputs = model(**tokens)
latency_ms = (time.time() - start) * 1000

logits = outputs.logits
print(f"Logits shape: {logits.shape}")
print(f"Latency: {latency_ms:.2f}ms")

print(f"\nTop 5 predicted next tokens:")
top5 = torch.topk(logits[0, -1, :], 5)
for score, token_id in zip(top5.values, top5.indices):
    print(f"  '{tokenizer.decode([token_id])}' (id={token_id.item()}, logit={score.item():.2f})")

# ---- STEP 3: Generation ----
print(f"\n--- GENERATION ---")
greedy = model.generate(**tokens, max_new_tokens=20, do_sample=False)
print(f"Greedy:  '{tokenizer.decode(greedy[0], skip_special_tokens=True)}'")

sampled = model.generate(**tokens, max_new_tokens=20, do_sample=True, temperature=0.1, top_p=0.9)
print(f"Sampled: '{tokenizer.decode(sampled[0], skip_special_tokens=True)}'")

print(f"\nSampled 3 times (notice variation):")
for i in range(3):
    out = model.generate(**tokens, max_new_tokens=15, do_sample=True, temperature=0.9)
    print(f"  Run {i+1}: '{tokenizer.decode(out[0], skip_special_tokens=True)}'")

print("\n" + "=" * 50)
print("DAY 1 COMPLETE")
print("=" * 50)