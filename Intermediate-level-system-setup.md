# 🚀 Intermediate Guide: Advanced AI Development on WSL

This guide is for users who have a basic understanding of AI development on WSL and want to enhance their setup with additional tools, optimized workflows, and GPU acceleration. 🎯

---
## 🔹 Step 1: Ensure Your System is Up to Date
```sh
sudo apt update && sudo apt upgrade -y
```
📌 **Why?** Keeps your system updated with the latest security patches and software versions.

---
## 🔹 Step 2: Install Additional Development Tools 🛠️
```sh
sudo apt install -y build-essential cmake git curl wget unzip htop tmux
```
📌 **Tools Installed:**
- `cmake` → Essential for compiling deep learning frameworks.
- `htop` → Monitors system performance.
- `tmux` → Terminal multiplexer for managing multiple sessions.

---
## 🔹 Step 3: Install Python & Virtual Environment (Recommended for Multiple Projects) 🐍
```sh
sudo apt install -y python3 python3-pip python3-venv
```
📌 **Why?**
- Ensures you have the latest Python version with package management tools.

### ✅ Setting Up a Dedicated AI Environment
```sh
mkdir ~/ai_workspace && cd ~/ai_workspace
python3 -m venv env
source env/bin/activate
```
📌 **Why?**
- Keeps dependencies isolated per project.

---
## 🔹 Step 4: Install Essential AI/ML Libraries 📚
```sh
pip install torch torchvision torchaudio transformers datasets numpy pandas matplotlib scikit-learn scikit-image seaborn tqdm
```
📌 **What’s Included?**
- `torch` → Deep learning framework.
- `transformers` → Hugging Face models.
- `scikit-image` → Image processing.
- `seaborn` → Advanced data visualization.
- `tqdm` → Progress bars for loops.

---
## 🔹 Step 5: Install Jupyter Notebook & VS Code Integration 📓
```sh
pip install jupyterlab
jupyter lab --no-browser --ip=0.0.0.0
```
📌 **Why?**
- Runs Jupyter remotely, allowing access from VS Code.

### ✅ Connect Jupyter to VS Code
1. Install [Remote - WSL Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl) in VS Code.
2. Open WSL in VS Code and run:
   ```sh
   jupyter notebook --generate-config
   ```
3. Configure remote access if needed.

---
## 🔹 Step 6: Enable GPU Support (If Available) ⚡

### ✅ Install NVIDIA CUDA & cuDNN
```sh
sudo apt install -y nvidia-cuda-toolkit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
📌 **Why?**
- Enables GPU acceleration for AI model training.

### ✅ Test CUDA Availability
```python
import torch
print(torch.cuda.is_available())
```

---
## 🔹 Step 7: Install and Configure Docker 🐳
```sh
sudo apt install -y docker.io
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```
📌 **Why?**
- Docker allows containerized AI development, useful for reproducibility and cloud deployment.

> **🔍 Verify Installation:**
```sh
docker --version
```

---
## 🔹 Step 8: Install MLflow for Experiment Tracking 🎯
```sh
pip install mlflow
```
📌 **Why?**
- Tracks machine learning experiments and manages models efficiently.

---
## 🎉 You're Now at an Intermediate Level! 🚀

Your setup now supports efficient AI/ML development with enhanced tools, GPU acceleration, and experiment tracking. You're ready to work on larger models and optimize workflows!

### 🔗 Additional Resources:
- [PyTorch CUDA Docs](https://pytorch.org/get-started/previous-versions/)
- [Docker for AI Development](https://docs.docker.com/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

Happy Coding! 💻🔥
