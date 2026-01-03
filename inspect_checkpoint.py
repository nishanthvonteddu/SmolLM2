from safetensors.torch import safe_open
import os

# Find the safetensors file (handles sharded or single-file)
model_dir = "smollm2_135m_instruct"
files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
files.sort()

print("Safetensor files found:")
for f in files:
    print("  ", f)

path = os.path.join(model_dir, files[0])

with safe_open(path, framework="pt") as f:
    keys = list(f.keys())

print("\nTotal tensors:", len(keys))
print("\nFirst 30 keys:")
for k in keys[:30]:
    print(k)

print("\nLast 10 keys:")
for k in keys[-10:]:
    print(k)
