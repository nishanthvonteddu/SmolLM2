import torch
from transformers import AutoTokenizer
from safetensors.torch import load_file

from config import SmolLM2Config
from model import SmolLM2ForCausalLM

@torch.no_grad()
def greedy_generate(model, input_ids, max_new_tokens=80):
    # input_ids: (1, T)
    for _ in range(max_new_tokens):
        logits = model(input_ids)              # (1, T, V)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (1, 1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("smollm2_135m_instruct", use_fast=True)

    cfg = SmolLM2Config()
    model = SmolLM2ForCausalLM(cfg).to(device).half()
    sd = load_file("smollm2_final_fp16.safetensors")
    model.load_state_dict(sd, strict=True)
    model.eval()

    prompt = "Summary: A girl and her dog went on an adventure."
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    out_ids = greedy_generate(model, input_ids, max_new_tokens=80)
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    print("=== PROMPT ===")
    print(prompt)
    print("\n=== OUTPUT ===")
    print(text)

if __name__ == "__main__":
    main()
