import os
import yaml
import shutil
import subprocess
import time
import psutil
from git import Repo, GitCommandError
from huggingface_hub import HfApi, HfFolder, create_repo

CONFIG_FILE = "config.yaml"

# Function to load YAML config file
def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"{CONFIG_FILE} not found. Creating a new one...")
        config = {
            'huggingface': {'username': 'dtzx', 'token': ''},
            'git': {'username': '', 'email': ''},
            'organization': 'TitanML',
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

# Function to track system resource usage
def track_resource_usage(start_time):
    cpu_usage = psutil.cpu_percent(interval=1)
    total_ram = psutil.virtual_memory().total / (1024 ** 3)  # in GB
    used_ram = psutil.virtual_memory().used / (1024 ** 3)  # in GB
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nTotal time taken: {elapsed_time:.2f} seconds")
    print(f"CPU usage: {cpu_usage}%")
    print(f"Total RAM: {total_ram:.2f} GB, Used RAM: {used_ram:.2f} GB")

# Function to prompt for Hugging Face login and update config
def update_huggingface_login(config):
    token = config['huggingface'].get('token', '')

    if not token:
        print("Please log in to Hugging Face (dtzx account).")
        token = input("Enter your Hugging Face API token (for dtzx): ").strip()
        config['huggingface']['token'] = token
        save_config(config)

    HfFolder.save_token(token)

# Function to prompt for Git login and update config
def update_git_login(config):
    username = config['git'].get('username', '')
    email = config['git'].get('email', '')

    if not username or not email:
        print("Please log in to Git (zhixitee GitHub account).")
        if not username:
            username = input("Enter your GitHub username (zhixitee): ").strip()
            config['git']['username'] = username

        if not email:
            email = input("Enter your GitHub email: ").strip()
            config['git']['email'] = email

        save_config(config)

    # Configure Git with username and email
    subprocess.run(["git", "config", "--global", "user.name", username], check=True)
    subprocess.run(["git", "config", "--global", "user.email", email], check=True)

# Function to clone and upload repos as model repositories
def clone_and_upload_hf_repo(config):
    api = HfApi()
    hf_username = config['huggingface']['username']
    hf_token = config['huggingface']['token']
    organization = config.get('organization', 'TitanML')

    if not hf_token:
        raise ValueError("Hugging Face token is missing. Please check your config file.")
    
    start_time = time.time()

    for hf_repo_url in config['repositories']:
        repo_name = hf_repo_url.split("/")[-1]
        cloned_repo_dir = f"cloned_{repo_name}"

        print(f"\nCloning repository: {hf_repo_url}")
        try:
            # Clone the repo
            Repo.clone_from(hf_repo_url, cloned_repo_dir)

            # Remove the .git directory from the cloned repo
            git_dir = os.path.join(cloned_repo_dir, ".git")
            if os.path.exists(git_dir):
                shutil.rmtree(git_dir)

            # Create a new model repository in the TitanML organization on Hugging Face
            new_repo_id = f"{organization}/{repo_name}"  # Construct repo_id as organization/repo_name
            print(f"Creating new model repository under organization {organization}...")
            repo_url = api.create_repo(repo_id=new_repo_id, exist_ok=True).git_url

            # Embed Hugging Face credentials into the remote URL for Git push
            remote_url_with_creds = repo_url.replace("https://", f"https://{hf_username}:{hf_token}@")

            # Initialize a new Git repository and push to Hugging Face
            os.chdir(cloned_repo_dir)
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "remote", "add", "origin", remote_url_with_creds], check=True)
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "populate repo"], check=True)
            subprocess.run(["git", "branch", "-M", "main"], check=True)
            subprocess.run(["git", "push", "-u", "origin", "main"], check=True)

        except (GitCommandError, Exception) as e:
            print(f"Error occurred while processing repository {hf_repo_url}: {e}")
        finally:
            # Clean up by deleting the cloned directory, even if an error occurs
            if os.path.exists(cloned_repo_dir):
                print(f"Cleaning up {cloned_repo_dir} due to error or completion.")
                shutil.rmtree(cloned_repo_dir)

    # Track resource usage and completion message
    track_resource_usage(start_time)
    print("Done cloning all repositories.")

# Main function
if __name__ == "__main__":
    # Load the config file
    config = load_config()

    # Update Hugging Face login (dtzx account)
    update_huggingface_login(config)

    # Update Git login (zhixitee GitHub account)
    update_git_login(config)

    # Clone and upload repositories
    clone_and_upload_hf_repo(config)
