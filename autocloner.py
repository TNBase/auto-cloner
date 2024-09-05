import os
import json
import shutil
import subprocess
import time
from git import Repo, GitCommandError
from huggingface_hub import HfApi, HfFolder, create_repo
import psutil  # To check system specs

CACHE_FILE = ".cache/repo_cache.json"

# Function to estimate time for cloning and uploading
def estimate_time_and_resources(repo_size, upload_speed):
    download_time = repo_size / (psutil.net_io_counters().bytes_recv + 1)  # in seconds
    upload_time = repo_size / (upload_speed + 1)  # in seconds
    
    return download_time, upload_time

# Function to check system resources (CPU, RAM)
def check_system_resources():
    cpu_usage = psutil.cpu_percent(interval=1)
    total_ram = psutil.virtual_memory().total / (1024 ** 3)  # in GB
    free_ram = psutil.virtual_memory().available / (1024 ** 3)  # in GB

    return cpu_usage, total_ram, free_ram

# Function to set up Git LFS
def setup_git_lfs(repo_dir):
    try:
        print("Initializing Git LFS...")
        subprocess.run(["git", "lfs", "install"], check=True)

        os.chdir(repo_dir)

        print("Configuring Git LFS to track large file types (*.bin, *.pt, *.ckpt)...")
        subprocess.run(["git", "lfs", "track", "*.bin", "*.pt", "*.ckpt"], check=True)

        subprocess.run(["git", "add", ".gitattributes"], check=True)
        print("Git LFS setup completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error setting up Git LFS: {e}")

# Function to create and load cache
def load_cache():
    if not os.path.exists(".cache"):
        os.makedirs(".cache")  # Ensure the .cache directory exists
    
    if not os.path.exists(CACHE_FILE):
        # If cache file doesn't exist, create it and return an empty cache
        with open(CACHE_FILE, "w") as f:
            json.dump({}, f)
        return {}
    
    try:
        # If the cache file exists, load it
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # If cache file is corrupted, print a message and reinitialize it
        print("Cache file is corrupted. Reinitializing cache...")
        with open(CACHE_FILE, "w") as f:
            json.dump({}, f)
        return {}

# Function to save cache
def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

# Function to clone and upload repos
def clone_and_upload_hf_repo(hf_repo_urls, org_name, cache):
    api = HfApi()
    
    if not HfFolder.get_token():
        raise ValueError("Please login to Hugging Face using `huggingface-cli login`.")
    
    for hf_repo_url in hf_repo_urls:
        repo_name = hf_repo_url.split("/")[-1]
        
        if hf_repo_url in cache:
            print(f"Skipping {hf_repo_url}, already processed.")
            continue
        
        print(f"Cloning repository: {hf_repo_url}")
        try:
            cloned_repo_dir = f"cloned_{repo_name}"
            start_time = time.time()

            # Clone the repo
            Repo.clone_from(hf_repo_url, cloned_repo_dir)

            # Setup Git LFS for large files
            setup_git_lfs(cloned_repo_dir)

            # Estimate time and system resources
            repo_size = sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(cloned_repo_dir) for f in files)
            upload_speed = psutil.net_io_counters().bytes_sent
            download_time, upload_time = estimate_time_and_resources(repo_size, upload_speed)
            cpu_usage, total_ram, free_ram = check_system_resources()

            print(f"Estimated download time: {download_time:.2f} seconds")
            print(f"Estimated upload time: {upload_time:.2f} seconds")
            print(f"Current CPU usage: {cpu_usage}%")
            print(f"Total RAM: {total_ram:.2f} GB, Free RAM: {free_ram:.2f} GB")

            # Remove the .git folder
            print("Removing .git directory...")
            git_dir = os.path.join(cloned_repo_dir, ".git")
            if os.path.exists(git_dir):
                shutil.rmtree(git_dir)

            # Create a new repository in the specified organization
            new_repo_full_name = f"{org_name}/{repo_name}"
            print(f"Creating new repository {new_repo_full_name}...")
            create_repo(repo_id=new_repo_full_name, organization=org_name, exist_ok=True)

            # Upload the files to the new repository
            print(f"Uploading files to {new_repo_full_name}...")
            api.upload_folder(
                folder_path=cloned_repo_dir,
                repo_id=new_repo_full_name,
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Repository {new_repo_full_name} successfully uploaded in {elapsed_time:.2f} seconds.")

            # Add repo to cache and save
            cache[hf_repo_url] = {"status": "uploaded"}
            save_cache(cache)

        except GitCommandError as e:
            print(f"Error occurred while cloning {hf_repo_url}: {e}")
        
        except OSError as e:
            print(f"File system error occurred: {e}. Possible reasons: low disk space, permissions issues.")
        
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        finally:
            if os.path.exists(cloned_repo_dir):
                shutil.rmtree(cloned_repo_dir)

        print(f"Finished processing {hf_repo_url}. Moving to the next one...\n")

# Main Function
if __name__ == "__main__":
    # Git Login Check
    try:
        print("Please ensure you're logged into Git.")
        subprocess.run(["git", "config", "--global", "credential.helper", "cache"], check=True)
    except subprocess.CalledProcessError:
        print("Git login failed. Please login using `git config --global user.name` and `git config --global user.email`.")
    
    # Load cache
    cache = load_cache()

    # Get user input
    org_name = input("Enter the Hugging Face organization name: ").strip()
    hf_repo_urls = input("Enter the repository URLs to clone (comma-separated): ").strip().split(',')

    # Remove extra spaces from each URL
    hf_repo_urls = [url.strip() for url in hf_repo_urls]

    # Run the clone and upload process
    clone_and_upload_hf_repo(hf_repo_urls, org_name, cache)

    print("All repositories processed.")
