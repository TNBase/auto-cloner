# Auto Cloner

Auto Cloner helps users to clone repositories and upload it their own/organisation's repository on huggingface. This code is made to automate the process of cloning LLM's repositories which could take couple of minutes to an hour depending on your hardware specifications. 

# Procedures

## Step 1

Clone this repository and run this code on terminal to login to your git and huggingface account

```bash
huggingface-cli login
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

## Step 2

Install all the necessary packages with the code below after navigating to the root directory of this project.

``` bash
pip install -r requirements.txt
```

## Step 3

To clone repositories on huggingface as well as uploading large files you need to run the code below on your terminal

``` bash
git lfs install
```

```bash
git lfs track "*.bin" "*.pit" "*.ckpt"
```

```bash
huggingface-cli lfs-enable-largefiles .
```

```bash
echo ".cache" >> .gitignore
```