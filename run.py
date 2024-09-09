import os
import shutil
import sys
from pathlib import Path

from loguru import logger

from utils import download_from_hf, push_to_hub


def download_and_upload_model(repo_name:str, local_download_model_dir:str, token: str) -> None:
    """
    TODO: 
    """
    local_model_dir = Path(local_download_model_dir) / Path(repo_name)
    model_name = repo_name.split("/")[-1]

    download_from_hf(repo_name=repo_name, local_model_dir=local_model_dir, file_name=None, token=token)
    push_to_hub(model_name=model_name, model_path=local_model_dir, token=token)
    # Clean up the models 
    shutil.rmtree(local_model_dir)



if __name__ == "__main__":
    # Inputs 
    repo_name_list = [
        "google/gemma-2-2b-it",
        "google/gemma-2-2b",
        "nvidia/Mistral-NeMo-Minitron-8B-Base"
    ]
    local_download_model_dir = Path("models")

    token = os.getenv("HF_TOKEN") 
    if token is None: 
        logger.warning("HF Token is not given. Please export HF_TOKEN to environment variable")
        sys.exit(1)

    failed_models = []

    for repo_name in repo_name_list:
        try:
            logger.info(f"Start model: {repo_name}")
            download_and_upload_model(
                repo_name=repo_name,
                local_download_model_dir=local_download_model_dir,
                token=token
            )
        except Exception as e: 
            logger.error(f"Error while downloading or uploading models {repo_name}. Detailed error: {e}")
            failed_models.append(repo_name)

    logger.info(f"All models has been processed. {len(failed_models)} models has failed. They are {failed_models}")