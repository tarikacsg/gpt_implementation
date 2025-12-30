# train_and_generate.py

import math
import torch
import torch.nn.functional as F

from dataset import create_dataloader_v1
from gpt_model import GPTModel, generate_with_kv_cache, count_parameters
import tiktoken  # only for prompt encoding/decoding in the demo


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()
        logits, _ = model(input_ids, past_kvs=None)  # (B, T, V)

        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.view(B * T, V),
            target_ids.view(B * T),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * B * T
        total_tokens += B * T

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def main():
    # Tiny config so it trains quickly. You can scale this up later.
    GPT_CONFIG_TINY = {
        "vocab_size": 50257,
        "context_length": 128,
        "emb_dim": 256,
        "n_heads": 4,
        "n_layers": 4,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)

    # Replace this with your own corpus file if you want
    sample_text = (
        "Hello, I am learning to build a GPT model from scratch. "
        "This is a tiny demo corpus for testing the training loop. "
    ) * 50

    dataloader, tokenizer = create_dataloader_v1(
        sample_text,
        batch_size=8,
        max_length=GPT_CONFIG_TINY["context_length"],
        stride=GPT_CONFIG_TINY["context_length"],
        shuffle=True,
    )

    model = GPTModel(GPT_CONFIG_TINY).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # A few epochs just as a demo
    for epoch in range(3):
        loss, ppl = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}: loss={loss:.4f}, ppl={ppl:.2f}")

    # ---- Generation demo ----
    model.eval()
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)

    out = generate_with_kv_cache(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=50,
        context_size=GPT_CONFIG_TINY["context_length"],
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print("\n=== Generated text ===\n")
    print(decoded_text)


if __name__ == "__main__":
    main()
