from dataclasses import dataclass

@dataclass
class SmolLM2Config:
    # Core architecture
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536

    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3

    # Positional / normalization
    max_position_embeddings: int = 8192
    rope_theta: float = 100000.0
    rms_norm_eps: float = 1e-5

    # Misc
    tie_word_embeddings: bool = True
