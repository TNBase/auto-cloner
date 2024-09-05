# Auto Cloner

**Auto Cloner** is a tool that automates the process of cloning repositories and uploading them to your personal or organization's repository on Hugging Face. This is particularly useful for cloning large language models (LLMs) repositories, which can take a significant amount of time depending on your system's hardware specifications.

## Features:
- Clone Hugging Face repositories.
- Set up Git LFS for large files.
- Automatically upload cloned repositories to your Hugging Face account or organization.
- Dynamically store Hugging Face and Git credentials for easy reuse.

---

## Prerequisites
Before proceeding, make sure you have the following:
- A Hugging Face account.
- A GitHub account (or any Git credentials configured).
- Python 3.x installed.

---

## Procedures

### Step 1: Clone this Repository and Login

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/auto-cloner.git
   cd auto-cloner
   ```
2. **Login to your Hugging Face account**
   ```bash
   huggingface-cli login
   ```
3. **Set up your Git credentials**
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your-email@example.com"
   ```
### Step 2: Install the Necessary Packages

1. **Install all required dependencies** by navigating to the root directory of the project and running:
   ```bash
   pip install -r requirements.txt
   ```
   This will install the following packages:
   - `huggingface_hub`:To interact with Hugging Face repositories.
   - `GitPython`:For cloning and pushing repositories.
   - `psutil`:For checking system resources.
   - `PyYAML`:To manage the configuration in YAML file.

### Step 3: Set Up Git LFS for Large Files

To enable large file uploads (e.g., model weights), you need to initialize Git LFS (Large File Storage):

1. **Install Git LFS**:
   ```bash
   git lfs install
   ```
2. **Track large file types commonly used in model repositories**:
   ```bash
   git lfs track "*.bin" "*.pit" "*.ckpt"
   ```
3. **Enable large file uploads on Hugging Face repositories**:
   ```bash
   huggingface-cli lfs-enable-largefiles .
   ```
### Step 4: Update `.gitignore`

To avoid tracking certain files such as cache and credentials, you should update your `.gitignore`:
```bash
echo ".cache" >> .gitignore
echo "config.yaml" >> .gitignore
```
This ensures sensitve or unnecessary data like the cache file and your configuration are not tracked by Git.

### Step 5: Edit the `config.yaml` (Optional)

After the first run, your Hugging Face token and Git credentials will be saved to `config.yaml`. You can also add a list of repositories you wish to clone in this file.

Here's an example of what your `config.yaml` might look like:
```bash
huggingface:
  token: "your-huggingface-token"

git:
  username: "your-git-username"
  email: "your-email@example.com"

repositories:
  - "https://huggingface.co/username/repo1"
  - "https://huggingface.co/username/repo2"
```
Make sure to update the `repositories` list with the repositories you want to clone.

### Step 6: Run the Auto Cloner Script

Once the configuration is set, you can run the Auto Cloner script to clone and upload repositories:

```bash
python auto_cloner.py
```

The script will:
- Prompt for your Hugging Face API token and Git credentials if they are not already in `config.yaml`.
- Clone each repository listed in `config.yaml`.
- Automatically upload the cloned content to a new repository in your Hugging Face account/organization.
- Clean up the cloned directory after each upload.

### Additional Notes

- **Git LFS**: Git Large File Storage (LFS) is essential for uploading large files such as model weights. Make sure you have initialized it correctly with `git lfs track`.
- **Caching**: The script uses a cache (`.cache/repo_cache.json`) to track already processed repositories, so they are not cloned or uploaded again.
- **Configuration**: Your Hugging Face API token and Git credentials are stored securely in `config.yaml `after the first login for ease of reuse.

### Troubleshooting

- **Missing Dependencies**: If any dependency is missing or outdated, run:
```bash
pip install -r requirements.txt --upgrade
```
- **Authentication Issues**: Make sure you are logged into both Hugging Face and Git. If issues persist, check your login credentials in `config.yaml`.