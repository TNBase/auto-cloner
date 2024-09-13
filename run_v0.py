import os
import shutil
import sys
from pathlib import Path
from loguru import logger
from utils import download_from_hf, push_to_hub
import asyncio
import psutil
from huggingface_hub import HfApi
from requests.exceptions import HTTPError

api = HfApi()

async def get_model_size(repo_name: str, token: str) -> float:
    """
    Fetches the total size of the model files in gigabytes from Hugging Face.
    """
    try:
        model_info = api.model_info(repo_name, token=token)
        total_size_bytes = sum(file.size for file in model_info.siblings if file.size is not None)
        total_size_gb = total_size_bytes / (1024 ** 3)  # Convert bytes to GB
        return total_size_gb
    except Exception as e:
        logger.error(f"Failed to retrieve size for {repo_name}: {e}")
        raise e

async def repo_exists(repo_name: str, token: str) -> bool:
    """
    Check if the repo already exists on Hugging Face.
    """
    try:
        api.repo_info(repo_id=repo_name.split("/")[-1], token=token)
        return True
    except HTTPError as e:
        if e.response.status_code == 404:
            return False
        else:
            logger.error(f"Error checking repo existence for {repo_name}: {e}")
            raise e

async def org_repo_exists(org_name: str, repo_name: str, token: str) -> bool:
    """
    Check if the repo already exists in the organization on Hugging Face.
    """
    try:
        repo_id = f"{org_name}/{repo_name}"
        api.repo_info(repo_id=repo_id, token=token)
        return True
    except HTTPError as e:
        if e.response.status_code == 404:
            return False
        else:
            logger.error(f"Error checking repo existence for {repo_id}: {e}")
            raise e

async def download_and_upload_model(repo_name: str, local_download_model_dir: str, token: str, org_name: str) -> None:
    """
    Downloads a model from Hugging Face, uploads it to a target repository, and deletes the local copy.
    """
    try:
        local_model_dir = Path(local_download_model_dir) / Path(repo_name)
        model_name = repo_name.split("/")[-1]

        # Check if the repo already exists in the organization
        if await org_repo_exists(org_name, model_name, token):
            logger.info(f"Repository {model_name} already exists in the organization {org_name}. Skipping download.")
            return

        # Skip download if model already exists locally
        if local_model_dir.exists():
            logger.warning(f"Model directory {local_model_dir} already exists. Skipping download.")
        else:
            model_size = await get_model_size(repo_name, token)
            logger.info(f"Starting download for {repo_name} (Estimated size: {model_size:.2f} GB)")
            download_from_hf(repo_name=repo_name, local_model_dir=local_model_dir, file_name=None, token=token)
            logger.info(f"Successfully downloaded model: {repo_name}")

        # Upload the model if repo doesn't already exist
        if await repo_exists(model_name, token):
            logger.info(f"Repository {model_name} already exists. Skipping creation.")
        else:
            logger.info(f"Creating repository {model_name}")
            api.create_repo(repo_id=model_name, token=token, repo_type="model")

        push_to_hub(model_name=model_name, model_path=local_model_dir, token=token)
        logger.info(f"Successfully uploaded model: {model_name}")

    except Exception as e:
        logger.error(f"Error during process for {repo_name}. Error: {e}")

    finally:
        # Ensure cleanup happens
        try:
            if local_model_dir.exists():
                logger.info(f"Cleaning up local model directory for {model_name}")
                shutil.rmtree(local_model_dir)
                logger.info(f"Successfully cleaned up local files for {model_name}")
        except OSError as cleanup_error:
            logger.error(f"Failed to clean up model directory {local_model_dir}. Error: {cleanup_error}")

def get_free_space_gb() -> float:
    """Returns the available disk space in gigabytes."""
    return psutil.disk_usage("/").free / (1024 ** 3)

async def process_model_queue(repo_name_queue, local_download_model_dir, token, org_name):
    """
    Processes the queue of models and ensures the disk space constraints are met.
    """
    active_downloads = []  # Initialize active downloads as a list
    total_space_to_use = 900  # The total space the script can use (900GB)
    min_free_space = 100  # Leave 100GB free at all times
    download_model_sizes = {}  # Store estimated model sizes to manage space

    while repo_name_queue:
        repo_name = repo_name_queue[0]
        free_space = get_free_space_gb()

        # Estimate model size using Hugging Face API
        try:
            model_size = await get_model_size(repo_name, token)
        except Exception as e:
            logger.error(f"Skipping model {repo_name} due to error: {e}")
            repo_name_queue.pop(0)
            continue

        total_active_space = sum(download_model_sizes.get(task, 0) for task in active_downloads)

        # Check for space and download
        if free_space - model_size >= min_free_space and (total_active_space + model_size) <= total_space_to_use:
            repo_name_queue.pop(0)
            logger.info(f"Starting download for {repo_name}. Estimated size: {model_size:.2f} GB")
            task = asyncio.create_task(download_and_upload_model(repo_name, local_download_model_dir, token, org_name))
            active_downloads.append(task)
            download_model_sizes[task] = model_size
        else:
            logger.info(f"Waiting for space... Current free space: {free_space:.2f} GB, Required: {model_size:.2f} GB")
            await asyncio.sleep(10)

        # Clean up finished tasks
        done, pending = await asyncio.wait(active_downloads, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
        for completed_task in done:
            del download_model_sizes[completed_task]  # Free up space used by completed task
        active_downloads = list(pending)  # Update the list of active downloads

    if active_downloads:
        logger.info("Waiting for all active downloads to complete...")
        await asyncio.gather(*active_downloads)

async def main(repo_name_list: list, local_download_model_dir: str, token: str, org_name: str):
    repo_name_queue = repo_name_list.copy()
    await process_model_queue(repo_name_queue, local_download_model_dir, token, org_name)

if __name__ == "__main__":
    # Inputs
    repo_name_list = [
        # "google/gemma-2-2b-it",
        # "google/gemma-7b-it",
        # "google/gemma2-27b-it",
        # "google/gemma-2b",
        # "google/gemma2-9b-it",
        # "NousResearch/Meta-Llama-3.1-70B",
        # "Qwen/Qwen-1.8B",
        # "NousResearch/Meta-Llama-3.1-8B",
        "NousResearch/Meta-Llama-3.1-70B-Instruct",
        # "Qwen/Qwen-1.8B-Chat",
        # "NousResearch/Meta-Llama-3.1-8B-Instruct",
        # "NousResearch/Meta-Llama-3.1-405B-FP8",
        # "Qwen/Qwen2-1.5B-Instruct",
        # "NousResearch/Llama-2-7b-hf",
        # "NousResearch/Llama-2-70b-chat-hf",
        # "Qwen/Qwen2-1.5B",
        # "NousResearch/Llama-2-13b-hf",
        # "mistralai/Mixtral-8x7B-v0.1",
        # "Qwen/Qwen2-0.5B-Instruct",
        # "mistralai/Mistral-7B-v0.1",
        # "mistral-community/Mixtral-8x22B-v0.1",
        # "Qwen/Qwen2-0.5B",
        # "mistralai/Mistral-7B-Instruct-v0.1",
        # "Qwen/Qwen-72B",
        # "Qwen/Qwen2-1.5B-Instruct-AWQ",
        # "mistralai/Mistral-7B-v0.3",
        "Qwen/Qwen-72B-Chat"
        # "Qwen/Qwen2-0.5B-Instruct-AWQ",
        # "mistralai/Mistral-7B-Instruct-v0.3",
        # "Qwen/Qwen-14B",
        # "Qwen/Qwen2-Math-1.5B",
        # "mistralai/Mistral-Nemo-Base-2407",
        # "Qwen/Qwen-14B-Chat",
        # "Qwen/Qwen2-72B-Instruct",
        # "Qwen/Qwen2-Math-1.5B-Instruct",
        # "mistralai/Mistral-Nemo-Instruct-2407",
        # "llava-hf/bakLlava-v1-hf",
        # "Qwen/Qwen-7B",
        # "Qwen/Qwen2-72B",
        # "Qwen/Qwen-7B-Chat",
        # "Qwen/Qwen2-72B-Instruct-AWQ",
        # "Qwen/Qwen2-beta-7B",
        # "Qwen/Qwen2-Math-72B",
        # "Qwen/Qwen2-beta-7B-Chat",
        # "Qwen/Qwen2-Math-72B-Instruct",
        # "Qwen/Qwen-7B-Chat-Int4",
        # "llava-hf/llava-1.5-13b-hf",
        # "Qwen/Qwen-7B-Chat-Int8",
        # "Qwen/Qwen2-7B-Instruct",
        # "Qwen/Qwen2-7B",
        # "Qwen/Qwen2-7B-Instruct-AWQ",
        # "Qwen/Qwen2-Math-7B",
        # "Qwen/Qwen2-Math-7B-Instruct",
        # "llava-hf/llava-1.5-7b-hf"
    ]
    local_download_model_dir = Path("models")

    # Get Hugging Face token from environment variables
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.warning("HF token is not provided. Please export HF_TOKEN to the environment variable.")
        sys.exit(1)

    # Get organization name from environment variables
    org_name = os.getenv("ORG_NAME")
    if not org_name:
        logger.warning("Organization name is not provided. Please export ORG_NAME to the environment variable.")
        sys.exit(1)

    asyncio.run(main(repo_name_list, local_download_model_dir, token, org_name))
