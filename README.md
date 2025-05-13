# 🎯 Pixel-wise A3C for Image Denoising

This project implements a **Reinforcement Learning (RL)**-based framework for grayscale image denoising using a **fully convolutional A3C (Asynchronous Advantage Actor-Critic)** model. Unlike traditional denoising methods, this approach learns to apply **pixel-level filters and fine-tuned actions** over multiple steps to iteratively clean noisy images.

---

## 🧠 Core Idea

We formulate image denoising as a **Markov Decision Process**:
- Each **pixel** is treated as an agent.
- The model takes the current image and a memory state, and outputs:
  - A **policy** (`π(a|s)`) indicating which filter or value change to apply per pixel.
  - A **value** (`V(s)`) estimating how clean the image is.
- The system evolves over time using **ConvGRU** for memory.

---

## 🧩 Key Files and Responsibilities

### 1. `train.py`
- Main script to run training and testing.
- Initializes model, agent, data loader, and environment.
- Trains over episodes using PSNR-based rewards.
- Saves denoised results and model weights periodically.

### 2. `MyFcn.py`
- Defines the model architecture:
  - Shared convolutional feature extractor.
  - Policy and value branches.
  - ConvGRU-like gating for updating hidden state.
- Takes a 65-channel tensor as input (1 grayscale + 64 hidden state channels).
- Outputs:
  - Action logits
  - Value map
  - Updated hidden state

### 3. `State.py`
- Simulates the image environment.
- Applies **pixel-level filters** like Gaussian, Median, Bilateral, and Box filters.
- Maintains and updates the internal image + GRU hidden state tensor.

### 4. `pixelwise_a3c.py`
- A3C Agent class:
  - Selects actions using the policy network.
  - Stores transitions (state, action, reward, value, logits).
  - Computes returns and updates the model using actor-critic loss.
  - Encourages exploration using entropy regularization.

### 5. `mini_batch_loader.py`
- Loads grayscale image data from the BSD68 dataset.
- Performs random augmentations during training:
  - Horizontal flips
  - Small rotations
  - Random crops
- Outputs image tensors in (B, C, H, W) format.

---

## 🗂 Directory Structure

pixelRL/
├── train.py
├── MyFcn.py
├── State.py
├── pixelwise_a3c.py
├── mini_batch_loader.py
├── BSD68/
│ └── gray/
├── training_BSD68.txt
├── testing.txt
├── resultimage_tf/
├── model_tf/
└── log_tf.txt


---

## ⚙️ Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- OpenCV

Install dependencies:

```bash
pip install tensorflow numpy opencv-python
