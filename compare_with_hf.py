import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import SmolLM2Config
from model import SmolLM2ForCausalLM
from safetensors.torch import load_file
import os


def load_ours(model_dir):
    cfg = SmolLM2Config()
    model = SmolLM2ForCausalLM(cfg)
    model.eval()

    path = os.path.join(model_dir, "model.safetensors")
    sd = load_file(path)
    model.load_state_dict(sd, strict=False)

    return model


def main():
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model_dir = "smollm2_135m_instruct"

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # input
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Hugging Face model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )
    hf_model.eval()

    # Your model
    my_model = load_ours(model_dir)

    with torch.no_grad():
        hf_logits = hf_model(**inputs).logits
        my_logits = my_model(inputs["input_ids"])

    # Compare
    diff = (hf_logits - my_logits).abs()
    print("HF logits shape:", hf_logits.shape)
    print("My logits shape:", my_logits.shape)
    print("Max absolute diff:", diff.max().item())
    print("Mean absolute diff:", diff.mean().item())


if __name__ == "__main__":
    main()
