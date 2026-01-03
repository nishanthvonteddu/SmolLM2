import os
import torch

from safetensors.torch import load_file
from config import SmolLM2Config
from model import SmolLM2ForCausalLM


def load_safetensors_state_dict(model_dir: str):
    # Prefer sharded files if present
    shards = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors") and "model-" in f])
    if shards:
        print(f"Found {len(shards)} shard(s). Loading and merging...")
        sd = {}
        for fname in shards:
            path = os.path.join(model_dir, fname)
            part = load_file(path)
            overlap = set(sd.keys()).intersection(part.keys())
            if overlap:
                raise RuntimeError(f"Overlapping keys between shards: {list(overlap)[:5]}")
            sd.update(part)
        return sd

    # Otherwise look for single-file model.safetensors
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        print("Found single model.safetensors. Loading...")
        return load_file(single)

    raise FileNotFoundError("No safetensors weights found in model_dir")


def main():
    model_dir = "smollm2_135m_instruct"

    cfg = SmolLM2Config()
    model = SmolLM2ForCausalLM(cfg)
    model.eval()

    print("Loading checkpoint state_dict...")
    sd = load_safetensors_state_dict(model_dir)

    # Load weights (strict=False because tied lm_head may cause one harmless mismatch)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    print("\n=== load_state_dict report ===")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    # Basic forward pass sanity check
    with torch.no_grad():
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)  # arbitrary tokens
        logits = model(input_ids)
        print("\nForward pass OK. Logits shape:", tuple(logits.shape))
        print("Logits stats: mean =", float(logits.mean()), "std =", float(logits.std()))


if __name__ == "__main__":
    main()
