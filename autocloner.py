import os
import shutil
import subprocess
import time
from git import Repo, GitCommandError
from huggingface_hub import HfApi, HfFolder, create_repo
import psutil  # To check system specs

# Function to estimate time for cloning and uploading
def estimate_time_and_resources(repo_size, upload_speed):
    # Estimation based on a basic bandwidth and system usage
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
        # Initialize Git LFS
        print("Initializing Git LFS...")
        subprocess.run(["git", "lfs", "install"], check=True)

        # Change directory to the cloned repository
        os.chdir(repo_dir)

        # Track common large file extensions
        print("Configuring Git LFS to track large file types (*.bin, *.pt, *.ckpt)...")
        subprocess.run(["git", "lfs", "track", "*.bin", "*.pt", "*.ckpt"], check=True)

        # Add .gitattributes to repository
        subprocess.run(["git", "add", ".gitattributes"], check=True)
        print("Git LFS setup completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error setting up Git LFS: {e}")

# Function to clone a repo, remove .git, and upload it to Hugging Face
def clone_and_upload_hf_repo(hf_repo_urls, org_name):
    api = HfApi()
    
    # Ensure Hugging Face login
    if not HfFolder.get_token():
        raise ValueError("Please login to Hugging Face using `huggingface-cli login`.")
    
    for hf_repo_url in hf_repo_urls:
        repo_name = hf_repo_url.split("/")[-1]
        print(f"Cloning repository: {hf_repo_url}")
        
        # Cloning the repository
        try:
            cloned_repo_dir = f"cloned_{repo_name}"
            start_time = time.time()

            # Clone the repo
            Repo.clone_from(hf_repo_url, cloned_repo_dir)

            # Setup Git LFS for large files in the cloned repository
            setup_git_lfs(cloned_repo_dir)

            # Estimate time and system resources
            repo_size = sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(cloned_repo_dir) for f in files)
            upload_speed = psutil.net_io_counters().bytes_sent  # Rough estimate of upload speed
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
        
        except GitCommandError as e:
            print(f"Error occurred while cloning {hf_repo_url}: {e}")
        
        except OSError as e:
            print(f"File system error occurred: {e}. Possible reasons: low disk space, permissions issues.")
        
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        finally:
            # Clean up
            if os.path.exists(cloned_repo_dir):
                shutil.rmtree(cloned_repo_dir)

        print(f"Finished processing {hf_repo_url}. Moving to the next one...\n")

# Main Function
if __name__ == "__main__":
    # Prompt user to login to Git
    try:
        print("Please ensure you're logged into Git. Logging in...")
        subprocess.run(["git", "config", "--global", "credential.helper", "cache"], check=True)
        subprocess.run(["git", "config", "--global", "user.name"], check=True)
    except subprocess.CalledProcessError:
        print("Git login failed. Please login using `git config --global user.name` and `git config --global user.email`.")
    
    # Input repository URLs and organization details
    hf_repo_urls = [
        "https://huggingface.co/username/repo1",
        "https://huggingface.co/username/repo2"
    ]
    org_name = "your-organization-name"

    clone_and_upload_hf_repo(hf_repo_urls, org_name)
