import torch
import transformers
from huggingface_hub import HfApi

print("PyTorch Version:", torch.__version__)
print("Transformers Version:", transformers.__version__)

api = HfApi()
print("Hugging Face connection successful!")
