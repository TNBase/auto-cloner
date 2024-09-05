import os
import yaml
import shutil
import subprocess
from git import Repo, GitCommandError
from huggingface_hub import HfApi, HfFolder, create_repo
import psutil

CONFIG_FILE = "config.yaml"

# Function to load YAML config file
def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"{CONFIG_FILE} not found. Creating a new one...")
        config = {
            'huggingface': {'token': ''},
            'git': {'username': '', 'email': ''},
            'repositories': []
        }
        save_config(config)
    else:
        with open(CONFIG_FILE, "r") as f:
            config = yaml.safe_load(f)
    return config

# Function to save YAML config file
def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f)

# Function to prompt for Hugging Face login and update config
def update_huggingface_login(config):
    token = config['huggingface'].get('token', '')

    if not token:
        print("Please log in to Hugging Face.")
        token = input("Enter your Hugging Face API token: ").strip()
        config['huggingface']['token'] = token
        save_config(config)

    HfFolder.save_token(token)

# Function to prompt for Git login and update config
def update_git_login(config):
    username = config['git'].get('username', '')
    email = config['git'].get('email', '')

    if not username or not email:
        print("Please log in to Git.")
        if not username:
            username = input("Enter your Git username: ").strip()
            config['git']['username'] = username

        if not email:
            email = input("Enter your Git email: ").strip()
            config['git']['email'] = email

        save_config(config)

    # Configure Git with username and email
    subprocess.run(["git", "config", "--global", "user.name", username], check=True)
    subprocess.run(["git", "config", "--global", "user.email", email], check=True)

# Function to clone and upload repos
def clone_and_upload_hf_repo(config):
    api = HfApi()
    
    hf_token = config['huggingface'].get('token', '')
    if not hf_token:
        raise ValueError("Hugging Face token is missing. Please check your config file.")
    
    for hf_repo_url in config['repositories']:
        repo_name = hf_repo_url.split("/")[-1]
        print(f"Cloning repository: {hf_repo_url}")
        try:
            cloned_repo_dir = f"cloned_{repo_name}"
            
            # Clone the repo
            Repo.clone_from(hf_repo_url, cloned_repo_dir)

            # Create a new repository in Hugging Face
            new_repo_full_name = f"{config['git']['username']}/{repo_name}"
            print(f"Creating new repository {new_repo_full_name}...")
            repo_url = api.create_repo(repo_id=new_repo_full_name, exist_ok=True).clone_url

            # Stage, commit, and push files to the new Hugging Face repository
            os.chdir(cloned_repo_dir)
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "remote", "add", "origin", repo_url], check=True)
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "populate repo"], check=True)
            subprocess.run(["git", "branch", "-M", "main"], check=True)
            subprocess.run(["git", "push", "-u", "origin", "main"], check=True)

        except GitCommandError as e:
            print(f"Error occurred while cloning {hf_repo_url}: {e}")
        finally:
            # Clean up by deleting the cloned directory
            if os.path.exists(cloned_repo_dir):
                shutil.rmtree(cloned_repo_dir)

# Main function
if __name__ == "__main__":
    # Load the config file
    config = load_config()

    # Update Hugging Face login
    update_huggingface_login(config)

    # Update Git login
    update_git_login(config)

    # Clone and upload repositories
    clone_and_upload_hf_repo(config)

    print("All repositories processed.")
