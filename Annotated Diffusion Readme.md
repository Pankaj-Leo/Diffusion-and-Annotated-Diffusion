Annotated DDPM: Denoising Diffusion from Scratch

This repository contains a modular, step-by-step implementation of **Denoising Diffusion Probabilistic Models (DDPM)**. Based on the seminal work by Ho et al. (2020), this project deconstructs the diffusion process into its core mathematical components and implements a high-performance **U-Net** backbone to handle image synthesis.

## 📖 Theoretical Overview

Diffusion models work through a two-stage stochastic process:

1. **Forward Diffusion ():** Gradually destroying data by adding Gaussian noise over  timesteps.
2. **Reverse Diffusion ():** A learned neural network that removes noise to recover the original data distribution.

---

## 🛠️ Technical Implementation

### 1. Forward Diffusion Process

The goal is to reach an isotropic Gaussian distribution at .

* **Variance Scheduling:** The code implement multiple schedules () to control noise levels:
* **Linear:** Standard schedule increasing from  to .
* **Cosine:** Proposed by Nichol et al. (2021) to prevent the data from becoming noise too quickly.


* **The "Nice Property":** Leveraging the fact that the sum of Gaussians is Gaussian, we sample  at any arbitrary timestep without iterative loops:



### 2. The Neural Network (Conditional U-Net)

To predict the noise  added at timestep , we use a specialized U-Net architecture:

* **Sinusoidal Time Embeddings:** Because the U-Net weights are shared across all , we inject time-step information using sinusoidal embeddings, allowing the model to distinguish between "coarse" and "fine" denoising phases.
* **Hybrid Attention Mechanisms:**
* **Standard Attention:** Captures global context in the bottleneck.
* **Linear Attention:** Used in higher-resolution blocks to maintain  memory efficiency.


* **Backbone Blocks:** Support for both standard **ResNet** blocks and modernized **ConvNeXT** blocks for improved gradient flow.

### 3. Training Objective

The model is trained to minimize the difference between the **actual noise** added and the **predicted noise**:



We utilize **Huber Loss** (Smooth L1) to provide robustness against outliers during the early stages of training.

### 4. Sampling & Inference

Inference follows the reverse Markov chain. Starting from pure noise , the model iteratively predicts and subtracts noise to reach .

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision einops datasets tqdm

```

### Usage

To train the model on Fashion-MNIST (28x28 images):

```python
from model import Unet, p_losses

# Initialize model
model = Unet(dim=28, channels=1, dim_mults=(1, 2, 4))
model.to(device)

# Training loop
for step, batch in enumerate(dataloader):
    t = torch.randint(0, T, (batch_size,), device=device).long()
    loss = p_losses(model, batch, t, loss_type="huber")
    loss.backward()

```

---

## 📈 Results

The model successfully generates diverse samples (e.g., T-shirts, shoes) from random noise after 5–10 epochs of training on localized datasets.

## 📚 References

* Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).
* Nichol, A., & Dhariwal, P. (2021). [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672).

---

**Would you like me to add a specific section on how to visualize the attention maps or implement the classifier-free guidance mentioned in the notebook?**