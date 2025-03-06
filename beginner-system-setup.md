# 🚀 Beginner's Guide: Setting Up AI Development on WSL

Welcome to the beginner-friendly guide on setting up your AI development environment on **Windows Subsystem for Linux (WSL)**! 🎯 This guide will help you install the necessary tools, packages, and dependencies to start building AI models and fine-tune them. 

---
## 🔹 Step 1: Install & Set Up WSL

### ✅ Enable WSL & Install Ubuntu
```sh
wsl --install -d Ubuntu
```
📌 **What this does?**
- Installs **WSL** with **Ubuntu** as the default Linux distribution.
- Ensures a seamless Linux environment within Windows.

> **🔍 Note:** If you already have WSL installed, update it using:
```sh
wsl --update
```

---
## 🔹 Step 2: Update & Upgrade Packages
```sh
sudo apt update && sudo apt upgrade -y
```
📌 **What this does?**
- `apt update` refreshes package lists.
- `apt upgrade -y` installs the latest updates for existing packages.

---
## 🔹 Step 3: Install Essential Development Tools 🛠️
```sh
sudo apt install -y build-essential git curl wget unzip
```
📌 **Tools Installed:**
- `build-essential` → C/C++ compilers & libraries.
- `git` → Version control system.
- `curl` & `wget` → Downloading files from the web.
- `unzip` → Extracting `.zip` files.

---
## 🔹 Step 4: Install Python & Virtual Environment 🐍
```sh
sudo apt install -y python3 python3-pip python3-venv
```
📌 **Why?**
- `python3` → Installs Python.
- `python3-pip` → Installs package manager for Python.
- `python3-venv` → Helps create isolated environments.

> **🔍 Check installation:**
```sh
python3 --version
pip3 --version
```

---
## 🔹 Step 5: Set Up a Virtual Environment 🎭
```sh
mkdir ~/ai_project && cd ~/ai_project
python3 -m venv venv
source venv/bin/activate
```
📌 **Why?**
- Creates a separate environment for your AI projects, avoiding conflicts.

> **🔍 To deactivate the environment:**
```sh
deactivate
```

---
## 🔹 Step 6: Install AI/ML Libraries 📚
```sh
pip install torch torchvision torchaudio transformers datasets numpy pandas matplotlib scikit-learn
```
📌 **What’s included?**
- `torch` → PyTorch framework for deep learning.
- `transformers` → Hugging Face library for LLMs.
- `datasets` → Dataset library from Hugging Face.
- `numpy, pandas` → Data processing.
- `matplotlib` → Visualization.
- `scikit-learn` → Machine learning utilities.

---
## 🔹 Step 7: Install Jupyter Notebook 📓
```sh
pip install jupyterlab
jupyter lab
```
📌 **Why?**
- Jupyter provides an interactive interface for coding & visualization.
- `jupyter lab` starts the notebook in a browser.

---
## 🔹 Step 8: Install GPU Support (If Available) ⚡

### ✅ Install NVIDIA CUDA for PyTorch
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
📌 **Why?**
- Enables GPU acceleration for AI model training.

> **🔍 Check CUDA availability:**
```python
import torch
print(torch.cuda.is_available())
```

---
## 🎉 You’re Ready! 🚀

You have successfully set up your AI development environment in WSL! 🎯 Now, you can start experimenting with AI models, fine-tuning LLMs, and exploring machine learning projects.

---
### 🔗 Additional Resources:
- [Hugging Face Docs](https://huggingface.co/docs)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Scikit-Learn Guide](https://scikit-learn.org/stable/user_guide.html)

Happy Coding! 💻🔥
