import torch
from torch.utils.data import IterableDataset

class StreamingTokenDataset(IterableDataset):
    """
    Streams text samples, tokenizes them, and yields fixed-length
    token sequences for causal LM training.
    """

    def __init__(self, hf_dataset, tokenizer, seq_len: int):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buffer = []

        for sample in self.dataset:
            text = sample["text"]

            # Tokenize without special tokens
            tokens = self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]

            buffer.extend(tokens)

            # Yield as many full sequences as possible
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)

                yield x, y
