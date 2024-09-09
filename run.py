import os
import shutil
from pathlib import Path

from utils import download_from_hf, push_to_hub

if __name__ == "__main__":
    repo_name = "facebook/opt-125m"
    model_name = repo_name.split("/")[-1]
    local_model_dir = Path("models") / repo_name
    
    token = os.getenv("HF_TOKEN")


    download_from_hf(repo_name=repo_name, local_model_dir=local_model_dir, file_name=None, token=token)

    push_to_hub(model_name=model_name, model_path=local_model_dir, token=token)

    # clean up
    shutil.rmtree(local_model_dir)
