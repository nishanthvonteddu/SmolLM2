import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SmolLM2Config


# -------------------------
# RMSNorm
# -------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (batch, seq, dim)
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


# -------------------------
# Rotary Embeddings (RoPE)
# -------------------------
def precompute_rope_freqs(dim, max_position, theta):
    """
    Returns cos/sin shaped (max_position, dim) to match HF RoPE application.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_position).float()
    freqs = torch.outer(positions, inv_freq)  # (max_position, dim/2)

    # HF uses cos/sin over a (max_position, dim) tensor by duplicating freqs
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_position, dim)

    return torch.cos(emb), torch.sin(emb)



def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin, position_ids):
    """
    HF-compatible RoPE
    x: (B, H, T, D)
    cos/sin: (max_seq_len, D)
    position_ids: (B, T)
    """
    cos = cos[position_ids].unsqueeze(1)  # (B, 1, T, D)
    sin = sin[position_ids].unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


# -------------------------
# Attention (GQA)
# -------------------------
class SmolLM2Attention(nn.Module):
    def __init__(self, cfg: SmolLM2Config):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.q_dim = self.num_heads * self.head_dim  # equals hidden_size

        # HF/LLaMA-style projections
        self.q_proj = nn.Linear(self.hidden_size, self.q_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.q_dim, self.hidden_size, bias=False)

        # Precompute RoPE caches (cos/sin)
        cos, sin = precompute_rope_freqs(
            dim=self.head_dim,
            max_position=cfg.max_position_embeddings,
            theta=cfg.rope_theta,
        )
        # register as buffers so they move with .to(device)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat K/V heads for GQA.
        x: (B, kv_heads, T, head_dim) -> (B, heads, T, head_dim)
        """
        if self.num_kv_heads == self.num_heads:
            return x
        repeat_factor = self.num_heads // self.num_kv_heads
        # repeat along head dimension
        return x.repeat_interleave(repeat_factor, dim=1)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, T, hidden)
        attn_mask: optional additive mask broadcastable to (B, heads, T, T)
                  where masked positions are large negative (e.g., -1e9)
        """
        B, T, _ = x.shape
        position_ids = torch.arange(T, device=x.device).unsqueeze(0)
        # Project
        q = self.q_proj(x)  # (B, T, q_dim=hidden)
        k = self.k_proj(x)  # (B, T, kv_dim)
        v = self.v_proj(x)  # (B, T, kv_dim)

        # Reshape to (B, heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)          # (B, H, T, D)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)       # (B, KvH, T, D)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)       # (B, KvH, T, D)

        # Apply RoPE to q and k
        q = apply_rope(q, self.rope_cos, self.rope_sin, position_ids)
        k = apply_rope(k, self.rope_cos, self.rope_sin, position_ids)


        # Expand k/v heads to match q heads
        k = self._repeat_kv(k)  # (B, H, T, D)
        v = self._repeat_kv(v)  # (B, H, T, D)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)
        attn_scores = attn_scores * (1.0 / math.sqrt(self.head_dim))

        # Causal mask (always)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal, float("-inf"))

        # Optional additional mask (e.g. padding)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)  # (B, H, T, D)

        # Back to (B, T, hidden)
        out = out.transpose(1, 2).contiguous().view(B, T, self.q_dim)
        out = self.o_proj(out)
        return out

# -------------------------
# MLP (SwiGLU)
# -------------------------
class SmolLM2MLP(nn.Module):
    def __init__(self, cfg: SmolLM2Config):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.intermediate_size = cfg.intermediate_size

        # LLaMA-style MLP: gate + up, then down
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# -------------------------
# Transformer Block
# -------------------------
class SmolLM2DecoderLayer(nn.Module):
    def __init__(self, cfg: SmolLM2Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = SmolLM2Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = SmolLM2MLP(cfg)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Attention block (pre-norm)
        x = x + self.self_attn(self.input_layernorm(x), attn_mask=attn_mask)
        # MLP block (pre-norm)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

# -------------------------
# Base model (like HF LlamaModel)
# -------------------------
class SmolLM2Model(nn.Module):
    def __init__(self, cfg: SmolLM2Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([SmolLM2DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        input_ids: (B, T)
        Returns: hidden_states (B, T, hidden)
        """
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        x = self.norm(x)
        return x


# -------------------------
# Causal LM (like HF LlamaForCausalLM)
# -------------------------
class SmolLM2ForCausalLM(nn.Module):
    def __init__(self, cfg: SmolLM2Config):
        super().__init__()
        self.cfg = cfg
        self.model = SmolLM2Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # Tie weights if requested
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.model(input_ids, attn_mask=attn_mask)
        logits = self.lm_head(hidden)
        return logits
