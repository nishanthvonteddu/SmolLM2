import torch
import gradio as gr
from transformers import AutoTokenizer
from safetensors.torch import load_file

from config import SmolLM2Config
from model import SmolLM2ForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "smollm2_135m_instruct",
    use_fast=True,
)

# Load model
cfg = SmolLM2Config()
model = SmolLM2ForCausalLM(cfg).to(device).half()
state_dict = load_file("smollm2_final_fp16.safetensors")
model.load_state_dict(state_dict, strict=True)
model.eval()

@torch.no_grad()
def generate(prompt, max_new_tokens=80):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(lines=3, label="Prompt"),
        gr.Slider(10, 200, value=80, step=10, label="Max new tokens"),
    ],
    outputs=gr.Textbox(label="Output"),
    title="SmolLM2-135M (From Scratch)",
    description="From-scratch PyTorch implementation, trained on TinyStories-Instruct",
)

if __name__ == "__main__":
    demo.launch()
