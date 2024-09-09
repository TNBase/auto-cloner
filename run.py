import os
import shutil
import sys
from pathlib import Path
from loguru import logger
from utils import download_from_hf, push_to_hub

def download_and_upload_model(repo_name: str, local_download_model_dir: str, token: str) -> None:
    """
    downloads a model from hugging face, uploads it to a target repository, and deletes the local copy.

    - includes error handling for the download and upload process.
    - logs the start and end of each operation, including cleanup.
    - ensures cleanup happens even if upload fails, to prevent storage issues.
    """
    try:
        local_model_dir = Path(local_download_model_dir) / Path(repo_name)
        model_name = repo_name.split("/")[-1]

        # step 1: download the model
        logger.info(f"starting download for {repo_name}")
        download_from_hf(repo_name=repo_name, local_model_dir=local_model_dir, file_name=None, token=token)
        logger.info(f"successfully downloaded model: {repo_name}")

        # step 2: upload the model
        logger.info(f"starting upload for {model_name}")
        push_to_hub(model_name=model_name, model_path=local_model_dir, token=token)
        logger.info(f"successfully uploaded model: {model_name}")

    except Exception as e:
        logger.error(f"error during the process for {repo_name}. detailed error: {e}")
        raise e  # re-raise the exception to be handled by the main loop

    finally:
        # cleanup step, even if there's an error, we attempt to clean up
        try:
            if local_model_dir.exists():
                logger.info(f"cleaning up local model directory for {model_name}")
                shutil.rmtree(local_model_dir)
                logger.info(f"successfully cleaned up local files for {model_name}")
        except Exception as cleanup_error:
            logger.error(f"failed to clean up model directory {local_model_dir}. error: {cleanup_error}")


if __name__ == "__main__":
    # inputs 
    repo_name_list = [
        "google/gemma-2-2b-it",
        "google/gemma-2-2b",
        "nvidia/Mistral-NeMo-Minitron-8B-Base"
    ]
    local_download_model_dir = Path("models")

    # get hugging face token from environment variables
    token = os.getenv("HF_TOKEN") 
    if token is None: 
        logger.warning("hf token is not provided. please export hf_token to the environment variable.")
        sys.exit(1)

    failed_models = []

    for repo_name in repo_name_list:
        try:
            logger.info(f"processing model: {repo_name}")
            download_and_upload_model(
                repo_name=repo_name,
                local_download_model_dir=local_download_model_dir,
                token=token
            )
        except Exception as e: 
            logger.error(f"failed to process model {repo_name}. error: {e}")
            failed_models.append(repo_name)

    # final logging of the summary
    if failed_models:
        logger.info(f"processing complete. {len(failed_models)} models failed: {failed_models}")
    else:
        logger.info("all models have been successfully processed.")
