## Day 1 — Feb 23, 2026

### AI Engineering
- A model is a math function, not magic: text → tokens → logits → text
- distilgpt2 has 81,912,576 parameters, takes 328MB RAM
- Logits shape [1, 5, 50257] — 50,257 scores for every possible next token
- Greedy = deterministic (same every run), Sampled = varies with temperature
- CPU latency was 639ms — GPU would be 5-20ms, 30-100x faster
- Low temperature on a weak model = repetitive garbage, not better quality

### Git
- 3 zones: working dir → staging (add) → local repo (commit) → remote (push)
- .gitignore protects from committing secrets, model weights, cache files
- core.autocrlf true handles Windows line endings automatically
- Convention: feat: prefix for new features in commit messages

### Surprised by
its also tokenize the numbers also the special characters the way they were tokenized 

### Question I have
I guess no questions all of my questions are answered.