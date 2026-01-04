import os
import time
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from config import SmolLM2Config
from model import SmolLM2ForCausalLM
from dataset import StreamingTokenDataset
from safetensors.torch import load_file


# ------------------------------------------------
# Logging setup
# ------------------------------------------------
def setup_logging():
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("logs/train.log"),
            logging.StreamHandler(),
        ],
    )


# ------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------
def save_checkpoint(path, model, optimizer, scheduler, step, best_loss):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "best_loss": best_loss,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"], ckpt["best_loss"]


# ------------------------------------------------
# Load pretrained weights
# ------------------------------------------------
def load_pretrained(model, model_dir):
    path = os.path.join(model_dir, "model.safetensors")
    sd = load_file(path)
    model.load_state_dict(sd, strict=False)


# ------------------------------------------------
# Text generation (monitoring only)
# ------------------------------------------------
@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt, max_new_tokens=80):
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    model.train()
    return text


# ------------------------------------------------
# Main training
# ------------------------------------------------
def main():
    setup_logging()

    device = torch.device("cuda")

    # -------------------------
    # Config
    # -------------------------
    cfg = SmolLM2Config()
    seq_len = 512
    batch_size = 2
    max_steps = 6000
    log_interval = 10
    gen_interval = 50
    warmup_steps = 20
    base_lr = 3e-4

    gen_prompt = "Summary: A girl and her dog went on an adventure."

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

    logging.info("Setup complete.")
    logging.info(f"Device: {device}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Seq len: {seq_len}")

    # -------------------------
    # Optimizer + Scheduler
    # -------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    # -------------------------
    # Checkpoints
    # -------------------------
    os.makedirs("checkpoints", exist_ok=True)
    latest_ckpt = "checkpoints/latest.pt"
    best_ckpt = "checkpoints/best.pt"

    step = 0
    best_loss = float("inf")

    if os.path.exists(latest_ckpt):
        logging.info(f"Resuming from checkpoint: {latest_ckpt}")
        step, best_loss = load_checkpoint(
            latest_ckpt, model, optimizer, scheduler
        )
        logging.info(f"Resumed at step={step}, best_loss={best_loss:.4f}")

    # -------------------------
    # Training loop
    # -------------------------
    start_time = time.time()

    for batch in train_loader:
        if step >= max_steps:
            break

        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        step += 1
        loss_val = loss.detach().item()
        lr = scheduler.get_last_lr()[0]

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            logging.info(
                f"step={step:04d} "
                f"loss={loss_val:.4f} "
                f"lr={lr:.2e} "
                f"time={elapsed:.1f}s"
            )

        if step % gen_interval == 0:
            logging.info("=== Generation sample ===")
            text = generate_sample(
                model, tokenizer, device, gen_prompt
            )
            logging.info(text)
            logging.info("=== End sample ===")

        save_checkpoint(
            latest_ckpt, model, optimizer, scheduler, step, best_loss
        )

        if loss_val < best_loss:
            best_loss = loss_val
            save_checkpoint(
                best_ckpt, model, optimizer, scheduler, step, best_loss
            )
            logging.info(
                f"New best model saved (loss={best_loss:.4f})"
            )


if __name__ == "__main__":
    main()
