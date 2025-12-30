# gpt_model.py

import math
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.shift


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    """
    Causal multi-head self-attention with optional KV cache.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x, b):
        # (B, T, d_out) -> (B, H, T, D)
        return x.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x, b):
        # (B, H, T, D) -> (B, T, d_out)
        return x.transpose(1, 2).contiguous().view(b, -1, self.d_out)

    def forward(self, x, past_kv=None):
        # x: (B, T_q, d_in)
        b, T_q, _ = x.shape

        k = self.W_key(x)    # (B, T_q, d_out)
        q = self.W_query(x)
        v = self.W_value(x)

        # KV cache: concat over time dimension if past exists
        if past_kv is not None:
            past_k, past_v = past_kv          # (B, T_past, d_out)
            k = torch.cat([past_k, k], dim=1) # (B, T_k, d_out)
            v = torch.cat([past_v, v], dim=1)

        present_kv = (k, v)

        # Split heads
        qh = self._split_heads(q, b)  # (B, H, T_q, D)
        kh = self._split_heads(k, b)  # (B, H, T_k, D)
        vh = self._split_heads(v, b)  # (B, H, T_k, D)

        T_k = kh.size(2)

        # Scaled dot-product attention
        attn_scores = qh @ kh.transpose(-2, -1)  # (B, H, T_q, T_k)

        # Build causal mask (T_q, T_k)
        q_pos = torch.arange(T_k - T_q, T_k, device=x.device)
        k_pos = torch.arange(T_k, device=x.device)
        causal = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # (T_q, T_k)
        attn_scores = attn_scores.masked_fill(
            ~causal.view(1, 1, T_q, T_k),
            float("-inf"),
        )

        attn_weights = torch.softmax(attn_scores / math.sqrt(self.head_dim), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = attn_weights @ vh  # (B, H, T_q, D)
        context = self._merge_heads(context, b)  # (B, T_q, d_out)
        context = self.out_proj(context)
        return context, present_kv


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, past_kv=None):
        # Attention block
        shortcut = x
        x_norm = self.norm1(x)
        att_out, present_kv = self.att(x_norm, past_kv=past_kv)
        x = shortcut + self.drop_shortcut(att_out)

        # Feed-forward block
        shortcut = x
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = shortcut + self.drop_shortcut(ff_out)

        return x, present_kv


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, past_kvs=None):
        """
        in_idx: (B, T)
        past_kvs: list of length n_layers, each element is (k, v) or None
        """
        B, T = in_idx.shape

        past_len = 0
        if past_kvs is not None and past_kvs[0] is not None:
            past_len = past_kvs[0][0].shape[1]

        tok_embeds = self.tok_emb(in_idx)  # (B, T, C)
        pos_ids = torch.arange(past_len, past_len + T, device=in_idx.device)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)  # (1, T, C)

        x = self.drop_emb(tok_embeds + pos_embeds)

        if past_kvs is None:
            past_kvs = [None] * len(self.trf_blocks)

        new_past_kvs = []
        for block, past_kv in zip(self.trf_blocks, past_kvs):
            x, present_kv = block(x, past_kv=past_kv)
            new_past_kvs.append(present_kv)

        x = self.final_norm(x)
        logits = self.out_head(x)  # (B, T, vocab)
        return logits, new_past_kvs


#####################################
# Decoding utilities
#####################################

def apply_top_k(logits, top_k):
    if top_k is None:
        return logits
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, top_k, dim=-1)
    min_values = values[..., -1, None]
    return torch.where(
        logits < min_values,
        torch.full_like(logits, float("-inf")),
        logits,
    )


def apply_top_p(logits, top_p):
    if top_p is None:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs > top_p
    # ensure at least one token remains
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    mask = torch.zeros_like(sorted_mask).scatter(-1, sorted_indices, sorted_mask)
    return logits.masked_fill(mask, float("-inf"))


def generate_with_kv_cache(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=1.0,
    top_k=None,
    top_p=None,
):
    """
    Autoregressive generation using KV cache.
    """
    model.eval()
    past_kvs = None

    for _ in range(max_new_tokens):
        # limit context to model's max
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            if past_kvs is None:
                logits, past_kvs = model(idx_cond, past_kvs=None)
            else:
                logits, past_kvs = model(idx[:, -1:], past_kvs=past_kvs)

        next_logits = logits[:, -1, :]  # (B, vocab)

        if temperature is not None and temperature > 0:
            next_logits = next_logits / temperature

        next_logits = apply_top_k(next_logits, top_k)
        next_logits = apply_top_p(next_logits, top_p)

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

        idx = torch.cat([idx, next_token], dim=1)

    return idx


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
