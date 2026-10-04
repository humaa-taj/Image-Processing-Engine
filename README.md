# Bloom 🌸

**Bloom** is an AI-powered image restoration and face-to-sketch web application that brings four generative computer-vision models together in one unified interface.

It can restore corrupted images, automatically route images to specialized restoration models, combine multiple restoration experts, and transform face photographs into artistic sketches.

## ✨ Features

Bloom provides four AI-powered workspaces:

| Workspace                   | Task | Description                                                                                                                                                |
| --------------------------- | ---: | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Universal Restoration**   |    1 | A convolutional denoising autoencoder restores images affected by salt-and-pepper noise, Gaussian blur, and rectangular occlusion.                         |
| **Hard-Routed Restoration** |    2 | A CNN classifier identifies the corruption type and routes the image to the appropriate specialist restoration model. Clean images use an identity bypass. |
| **Soft Mixture-of-Experts** |    3 | A learned gating network continuously combines identity and specialist restoration experts using soft routing weights.                                     |
| **Face-to-Sketch**          |    4 | A style-conditioned pix2pix cGAN converts face photographs into sketches using a U-Net generator and PatchGAN discriminator.                               |

### What you can do

* Upload JPG or PNG images
* Use bundled sample images
* Apply different image corruptions and severity levels
* Restore corrupted images
* Generate face sketches in multiple styles
* View restoration error maps
* Inspect classifier probabilities
* View Soft-MoE routing weights
* Monitor inference time
* Download generated results
* Check model availability through the System page

---

## 🧠 Machine Learning Pipeline

Bloom uses the following architecture:

```text
PyTorch
   ↓
Model Training
   ↓
ONNX Export
   ↓
ONNX Runtime
   ↓
FastAPI Backend
   ↓
React + Tailwind Frontend
```

### Technology Stack

**Machine Learning**

* PyTorch
* Torchvision
* ONNX
* ONNX Runtime
* Optuna
* MLflow

**Backend**

* Python
* FastAPI
* Uvicorn

**Frontend**

* React
* Vite
* Tailwind CSS
* Nginx

**Deployment**

* Docker
* Docker Compose

---

# 🚀 Run Bloom with Docker

## Requirements

You only need:

* Git
* Docker
* Docker Compose

Python and Node.js are **not required** when running Bloom with Docker.

## 1. Clone the repository

```bash
git clone https://github.com/humaa-taj/Image-Processing-Engine.git
cd Image-Processing-Engine
```

## 2. Download the Face-to-Sketch model

The Face-to-Sketch generator is approximately 168 MB and is distributed separately from the Git repository.

Download it into:

```text
models/onnx/task4_generator.onnx
```

### Linux / macOS

```bash
curl -L -o models/onnx/task4_generator.onnx https://github.com/FaatehHaneef/Image-Processing-Engine/releases/download/models-v1/task4_generator.onnx
```

### Windows PowerShell

```powershell
curl.exe -L -o models/onnx/task4_generator.onnx https://github.com/FaatehHaneef/Image-Processing-Engine/releases/download/models-v1/task4_generator.onnx
```

Without this model, the three image-restoration workspaces can still be used, but Face-to-Sketch will report that its model is unavailable.

## 3. Start Bloom

```bash
docker compose up --build
```

Once the containers are running, open:

**http://localhost:8080**

The application displays **MODELS LOADED** when all required models are available.

To stop Bloom:

```bash
Ctrl+C
```

Or:

```bash
docker compose down
```

---

# 🖼️ Bloom Workspaces

## Universal Restoration

The Universal Restoration model uses a single convolutional autoencoder to restore several types of image corruption:

* Salt-and-pepper noise
* Gaussian blur
* Rectangular occlusion

The model uses a compressed **8×8 latent bottleneck** and does not use skip connections.

---

## Hard-Routed Restoration

The Hard-Routed system first classifies the type of corruption.

```text
Input Image
     ↓
CNN Classifier
     ↓
Corruption Type
 ┌───┼──────────────┐
 ↓   ↓              ↓
Noise Blur      Occlusion
 ↓   ↓              ↓
Specialist Autoencoder
     ↓
Restored Image
```

Exactly one specialist model is selected for each corrupted image.

Clean images use an identity bypass.

---

## Soft Mixture-of-Experts

The Soft-MoE model uses a learned gate to continuously combine several experts.

```text
                 ┌── Identity
                 ├── Noise Expert
Input → Gate ────┼── Blur Expert
                 └── Occlusion Expert
                       ↓
                 Weighted Output
```

Unlike hard routing, multiple experts can contribute to the final restoration.

The gate is initialized using the classifier and fine-tuned end-to-end.

---

## Face-to-Sketch

The Face-to-Sketch workspace uses a style-conditioned **pix2pix cGAN**.

Architecture:

```text
Face Image
    ↓
U-Net Generator
    ↓
Sketch
    ↑
PatchGAN Discriminator
```

The model was trained using the **FS2K** dataset and supports three sketch styles.

---

# 📁 Repository Structure

```text
Image-Processing-Engine/
│
├── src/
│   └── Shared training code, models, losses and metrics
│
├── scripts/
│   └── Data preparation, training, evaluation and ONNX export
│
├── configs/
│   └── Final hyperparameters selected through Optuna
│
├── artifacts/
│   ├── splits/
│   ├── manifests/
│   ├── optuna/
│   └── results/
│
├── models/
│   └── onnx/
│
├── backend/
│   ├── app/
│   ├── samples/
│   ├── tests/
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── Dockerfile
│
├── docs/
│
├── tests/
│
├── docker-compose.yml
│
└── README.md
```

---

# 🔬 Training and Evaluation

Training is performed locally using Python and a GPU. Docker is primarily used to serve the trained models and run the application.

## Python Environment

Python 3.11 is recommended.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install the appropriate PyTorch version for your CUDA environment.

---

# 📊 Datasets

Bloom uses two primary datasets.

### Oxford-IIIT Pet Dataset

Used for image restoration experiments.

The dataset should be placed under:

```text
data/oxford-iiit-pet/
```

with:

```text
images/
annotations/
```

Dataset:

https://www.robots.ox.ac.uk/~vgg/data/pets/

### FS2K

Used for face-to-sketch generation.

Expected structure:

```text
data/fs2k/FS2K/
├── photo/
├── sketch/
├── anno_train.json
└── anno_test.json
```

Dataset repository:

https://github.com/DengPingFan/FS2K

---

# 🧪 Experiments

Bloom uses **Optuna** for hyperparameter optimization and **MLflow** for experiment tracking.

Start MLflow with:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

---

# 🛠️ Development Without Docker

For local development, start the backend:

```bash
python -m uvicorn backend.app.main:app --port 8000
```

Then start the frontend in another terminal:

```bash
cd frontend
npm install
npm run dev
```

The development frontend will be available at:

```text
http://localhost:5173
```

---

# 📈 Model Export

After training, models can be exported to ONNX:

```bash
python scripts/export_onnx.py --all
```

Verify the exported models with:

```bash
python scripts/verify_onnx.py --all
```

---

# 🧪 Tests

Run the test suite with:

```bash
python -m pytest
```

---

# 📚 Attribution

### Oxford-IIIT Pet Dataset

O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, **"Cats and Dogs"**, CVPR 2012.

License: CC BY-SA 4.0.

### FS2K

D.-P. Fan et al., **"Facial-Sketch Synthesis: A New Challenge"**, Machine Intelligence Research, 2022.

Code is released under the MIT license.

---

# 🌸 About Bloom

Bloom brings multiple image-generation and restoration techniques into a single accessible application.

From removing image corruption to transforming faces into artistic sketches, the goal is to make advanced computer-vision models easy to experiment with through one unified interface.

**Train → Export → Serve → Create.**
