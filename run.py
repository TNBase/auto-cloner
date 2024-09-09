import os
import shutil
import sys
from pathlib import Path
from loguru import logger
from utils import download_from_hf, push_to_hub

def download_and_upload_model(repo_name: str, local_download_model_dir: str, token: str) -> None:
    """
    downloads a model from hugging face, uploads it to a target repository, and deletes the local copy.
    """

    try:
        local_model_dir = Path(local_download_model_dir) / Path(repo_name)
        model_name = repo_name.split("/")[-1]

        # check if model already exists locally
        if local_model_dir.exists():
            logger.warning(f"Model directory {local_model_dir} already exists. Skipping download.")
        else:
            # download the model
            logger.info(f"Starting download for {repo_name}")
            download_from_hf(repo_name=repo_name, local_model_dir=local_model_dir, file_name=None, token=token)
            logger.info(f"Successfully downloaded model: {repo_name}")

        # upload the model
        logger.info(f"Starting upload for {model_name}")
        push_to_hub(model_name=model_name, model_path=local_model_dir, token=token)
        logger.info(f"Successfully uploaded model: {model_name}")

    except (OSError, ValueError) as e:  # Example of more specific exception handling
        logger.error(f"Error during the process for {repo_name}. Detailed error: {e}")
        raise e  # re-raise the exception to be handled by the main loop

    finally:
        # Ensure cleanup happens even if there's an error
        try:
            if local_model_dir.exists():
                logger.info(f"Cleaning up local model directory for {model_name}")
                shutil.rmtree(local_model_dir)
                logger.info(f"Successfully cleaned up local files for {model_name}")
        except OSError as cleanup_error:
            logger.error(f"Failed to clean up model directory {local_model_dir}. Error: {cleanup_error}")


if __name__ == "__main__":
    # Inputs
    repo_name_list = [
        "google/gemma-2-2b-it",
        "Qwen/Qwen2-1.5B"
    ]
    local_download_model_dir = Path("models")

    # Get Hugging Face token from environment variables
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.warning("HF token is not provided. Please export HF_TOKEN to the environment variable.")
        sys.exit(1)

    failed_models = []

    for repo_name in repo_name_list:
        try:
            logger.info(f"Processing model: {repo_name}")
            download_and_upload_model(
                repo_name=repo_name,
                local_download_model_dir=local_download_model_dir,
                token=token
            )
        except Exception as e:
            logger.error(f"Failed to process model {repo_name}. Error: {e}")
            failed_models.append(repo_name)

    # Final logging of the summary
    if failed_models:
        logger.info(f"Processing complete. {len(failed_models)} models failed: {failed_models}")
    else:
        logger.info("All models have been successfully processed.")
