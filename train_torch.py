
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

from config import SmolLM2Config
from model import SmolLM2ForCausalLM
from dataset import StreamingTokenDataset
from safetensors.torch import load_file


def load_pretrained(model, model_dir):
    path = os.path.join(model_dir, "model.safetensors")
    sd = load_file(path)
    model.load_state_dict(sd, strict=False)


def main():
    device = torch.device("cuda")

    # -------------------------
    # Config
    # -------------------------
    cfg = SmolLM2Config()
    seq_len = 512
    batch_size = 2

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        "smollm2_135m_instruct",
        use_fast=True,
    )

    # -------------------------
    # Dataset
    # -------------------------
    hf_ds = load_dataset(
        "roneneldan/TinyStoriesInstruct",
        split="train",
        streaming=True,
    )

    train_ds = StreamingTokenDataset(
        hf_dataset=hf_ds,
        tokenizer=tokenizer,
        seq_len=seq_len,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=0,
    )

    # -------------------------
    # Model
    # -------------------------
    model = SmolLM2ForCausalLM(cfg).to(device)
    load_pretrained(model, "smollm2_135m_instruct")
    model.train()

    print("Setup complete.")
    print("Device:", device)
    print("Batch size:", batch_size)
    print("Seq len:", seq_len)


if __name__ == "__main__":
    main()
