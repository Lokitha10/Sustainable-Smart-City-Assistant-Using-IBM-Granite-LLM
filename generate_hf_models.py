from huggingface_hub import snapshot_download
import os

model_path = snapshot_download(
    repo_id="ibm-granite/granite-3.3-2b-instruct",
    cache_dir="hf_models/ibm-granite-3.3-2b-instruct",
    token=os.getenv("HF_TOKEN")
)

print("Model downloaded to:", model_path)
