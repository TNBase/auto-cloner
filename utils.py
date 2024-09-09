"""Utils functions for loading models from huggingfacehub."""

# ───────────────────────────────────────────────────── imports ────────────────────────────────────────────────────── #
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem, create_repo, snapshot_download
from loguru import logger


def download_from_hf(
    repo_name: str, local_model_dir: Path, file_name: str | None = None, token: str | None = None
) -> str:
    """Download an entire repo from Huggingface, to avoid loading the model into RAM.

    Note: we only download pytorch models (ex. .bin files) if they are available.
    we download .safetensors files when there is only safetensors available.

    We do this to avoid download extra files that we dont need. (ex. msgpack, h5, etc.)

    Args:
        repo_name (str): model name / repo name
        local_model_dir (Path): local model directory
        token (str, optional): huggingface token. Defaults to None.

    Returns:
        str: location of the model
    """
    fs = HfFileSystem(token=token)

    # List all the files in the directory
    files = fs.ls(repo_name, detail=False)

    # check for an index.json file type and use that type
    file_indexes = [f for f in files if f.endswith(".index.json")]

    # are we dealing with a distributed download, in which case only download the copy with the .index.json type
    multi_file = len(file_indexes) > 0
    logger.debug("index.json files: ", file_indexes)
    if multi_file:
        # are there any .bin.index.json files
        any_bin_files = any(f for f in file_indexes if f.endswith("bin.index.json"))
        any_pt_files = any(f for f in file_indexes if f.endswith("pt.index.json"))
        any_pth_files = any(f for f in file_indexes if f.endswith("pth.index.json"))

        available_torch_files = any_bin_files or any_pt_files or any_pth_files

        # are there any .safetensors.index.json files
        available_safetensors_files = any(f for f in file_indexes if f.endswith("safetensors.index.json"))
        logger.debug("Torch model files available: ", available_torch_files)
        logger.debug("Safetensors files available: ", available_safetensors_files)

    else:
        # check if there are .torch files
        available_torch_files = any(f for f in files if (f.endswith(".bin") or f.endswith(".pt") or f.endswith(".pth")))

        # check if there are .safetensors files:
        available_safetensors_files = any(f for f in files if f.endswith(".safetensors"))

    # use pytorch if it is available, if it isnt then assume .safetensors is available
    # e.g. this is the case with llama 2.
    safe_only = available_safetensors_files and (not available_torch_files)

    if safe_only:
        snapshot_patterns = [
            "*.msgpack",
            "*.h5",
            "coreml/**/*",
            "*.tflite",
            "*.onnx",
            "*.bin",
            "*.pt",
            "*.pth",
            "*.gguf",
        ]
    else:
        snapshot_patterns = [
            "*.msgpack",
            "*.h5",
            "*.safetensors",
            "coreml/**/*",
            "*safetensors*",
            "*.ot",
            "*.tflite",
            "*.onnx",
            "*.gguf",
        ]

    # if file_name is given, then only download that file.
    allow_patterns = None
    if file_name is not None:
        allow_patterns = [file_name]

    marker_path = local_model_dir / "download_in_progress_marker"

    local_model_dir.mkdir(parents=True, exist_ok=True)
    # Open the file in write mode, which creates the file if it doesn't exist
    with open(marker_path, "w") as _:
        pass
        # Since we want to create an empty file, we don't write anything
    logger.info("Beginning download")
    location = snapshot_download(
        repo_id=repo_name,
        local_dir=str(local_model_dir),
        local_dir_use_symlinks=False,  # force the load to be in the volume mount, not in cache.
        allow_patterns=allow_patterns,
        ignore_patterns=snapshot_patterns,
        token=token,
        resume_download=True,
    )

    marker_path.unlink()

    return location


def push_to_hub(model_name: str, model_path: str, token: str) -> None:
    """Push a model to the Hugging Face Hub

    Args:
        model_name (str): model name. Should be "opt-125m", not "facebook/opt-125m"
        model_path (str): path to the local model directory
        token (str): HF token
    """
    hub_id = f"TitanML/{model_name}"
    api = HfApi(token=token)

    create_repo(hub_id, token=token, repo_type="model")

    api.upload_folder(
        folder_path=model_path,
        repo_id=hub_id,
        repo_type="model",
    )
    logger.info(f"Model pushed to Hugging Face Hub: {hub_id}")
