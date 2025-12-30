# Mini GPT From Scratch (with KV Cache + Sampling)

This repo implements a **small GPT-style language model from scratch in PyTorch**, with:

- A simple **dataset + dataloader** for next-token prediction  
- A **Transformer-based GPT model** (pre-LN)  
- **KV-cache–based autoregressive decoding**  
- **Temperature / top-k / top-p (nucleus) sampling**  
- A minimal **training loop** and **generation demo**

It’s designed for learning and experimentation, not for production-scale training.

---

## Project Structure

```text
.
├── dataset.py              # Sliding-window dataset + dataloader
├── gpt_model.py            # GPT model, Transformer blocks, KV cache, decoding utils
└── train_and_generate.py   # Training loop + text generation demo
