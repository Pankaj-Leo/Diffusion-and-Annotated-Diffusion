

# 🚗 PyTorch Generative Diffusion: Automotive Design from Scratch

A high-performance, bottom-up implementation of **Denoising Diffusion Probabilistic Models (DDPM)** using PyTorch. This project demonstrates the mathematical and architectural foundations of generative AI by training a model to generate realistic vehicle silhouettes from pure Gaussian noise using the **Stanford Cars Dataset**.

## 📌 Project Overview

Diffusion models are at the forefront of the generative AI revolution (powering tools like Stable Diffusion and Midjourney). This repository breaks down the complexity of DDPMs into a modular, documented PyTorch implementation. It covers the entire lifecycle: from the mathematical forward diffusion process to a time-aware U-Net for the reverse denoising phase.

### 🛠 Technical Highlights

* **Mathematical Foundation:** Implements the linear variance schedule () to systematically add Gaussian noise across  timesteps.
* **Architecture:** Custom **U-Net** backbone featuring symmetric Downsampling/Upsampling blocks with residual skip connections to preserve spatial integrity.
* **Time-Step Awareness:** Integration of **Sinusoidal Positional Embeddings** that allow the network to adapt its denoising logic based on the specific level of corruption.
* **Training Objective:** Optimization via **Mean Squared Error (MSE)** loss, predicting the added noise  rather than the clean image directly.

---

## 🏗 Model Architecture & Design

The model is built on a specialized U-Net architecture designed to understand the geometry of vehicles.

| Hyperparameter | Value |
| --- | --- |
| **Dataset** | Stanford Cars (~8,000 images) |
| **Image Resolution** | 64 x 64 |
| **Diffusion Steps (T)** | 1,000 (typical) |
| **Optimizer** | Adam (Learning Rate: 2e-4) |
| **Loss Function** | MSE Loss |

### The Two-Phase Process:

1. **Forward Diffusion ():** Gradually destroys information by transitioning an image from structure to high entropy (Gaussian noise).
2. **Reverse Diffusion ():** The trained U-Net acts as an oracle, predicting the noise at each step to "recover" a clean vehicle image from static.

---

## 💻 Code Insights: Forward Sampling

The following snippet represents the core logic for the forward noise scheduler:

```python
def forward_diffusion_sample(x_0, t, device="cpu"):
    noise = torch.randn_like(x_0)
    # Extracts pre-calculated alpha values based on timestep t
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    
    # Mathematical transformation: x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*noise
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

```

---

## 🚀 Business & Industry Application

Why implement Diffusion for automotive data?

1. **Rapid Prototyping:** Designers can generate thousands of unique aerodynamic concepts in seconds to inspire new model silhouettes.
2. **Synthetic Data Generation:** High-fidelity car images can be used to train Computer Vision models for Autonomous Vehicles (AV), simulating rare vehicle types or edge-case scenarios.
3. **Customer Personalization:** Enabling real-time visual modification tools for luxury car configurators.

---

## 🛠 Setup and Usage

1. **Clone the Repository:**



2. **Install Dependencies:**
```bash
pip install torch torchvision matplotlib tqdm

```


3. **Run the Notebook:**
Launch `diffusion_model.ipynb` to view the training pipeline and sampling results.

## 🤝 Acknowledgments

* *Denoising Diffusion Probabilistic Models* (Ho et al., 2020).
* *Diffusion Models Beat GANs on Image Synthesis* (Dhariwal & Nichol, 2021).
* Stanford Cars Dataset and HuggingFace Annotated Diffusion.