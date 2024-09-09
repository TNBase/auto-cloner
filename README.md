# Auto Cloner

Auto Cloner is a tool designed to automate the process of downloading models from Hugging Face and uploading them to your own Hugging Face repository. It simplifies the workflow for managing large model files, automating both the download and upload, and cleaning up afterward.

## Features

- Download models from Hugging Face.
- Upload the downloaded models to your Hugging Face repository.
- Automatically cleans up downloaded models after successful upload

## Prerequisites

Before using Auto Cloner, ensure you have the following:

- Python 3.x installed.
- A Hugging Face account with a valid API token.
- Git LFS (Large File Storage) installed for handling large model files.

### Step 1: Clone the Repository

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/auto-cloner.git
   cd auto-cloner
   ```

### Step 2: Install Dependencies
2. Install the required Python packages:
   ``` bash
   pip install -r requirements.txt
   ```

### Step 3: Export Your Hugging Face API Token
3. To authenticate with Hugging Face, you'll need to export your Hugging Face API token as an environment variable:
   ```bash 
   export HF_TOKEN=<your_hugging_face_token>
   ```

### Step 4: Run the Script
4. Run the script using Python 3:
   ```bash
   python3 run.py
   ```