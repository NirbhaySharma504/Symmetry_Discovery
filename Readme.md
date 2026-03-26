# Symmetry Discovery

A research-oriented implementation for discovering and learning symmetries from data using deep learning. This project was developed as part of a GSoC selection task focused on symmetry-aware machine learning and physics-inspired representation learning.

---

## Overview

This project explores how neural networks can **learn transformations (symmetries)** directly from data rather than relying on predefined rules.

Using MNIST as a controlled setting, the pipeline:

- Generates rotated datasets
- Learns latent representations using a Variational Autoencoder (VAE)
- Models transformations in latent space
- Discovers symmetries in both supervised and unsupervised settings

---

## Tasks Completed

### Task 1: Dataset Preparation & Latent Space

- Created a rotated MNIST dataset (0°–330° in 30° steps)
- Optional filtering for digits (e.g., 0–1 for efficiency)
- Built and trained a **Variational Autoencoder (VAE)**
- Encoded images into a structured latent space

### Task 2: Supervised Symmetry Discovery

- Trained an **MLP on latent space**
- Learned mappings corresponding to rotation transformations
- Modeled symmetry as transformations between latent vectors

### Task 3: Unsupervised Symmetry Discovery

- Learned symmetry operators directly from latent representations
- Identified transformations that preserve class/logit structure
- Demonstrated emergence of rotation as a discovered symmetry

---

## Key Components

### Models

- `VAE` – Learns compact latent representation
- `RotationMLP` – Learns explicit transformation mappings
- `LatentClassifier` – Evaluates latent separability
- `SymmetryGenerator` – Learns symmetry operators

### Pipeline

1. Load & rotate dataset
2. Train VAE
3. Encode dataset → latent space
4. Train transformation models
5. Evaluate symmetry discovery

---

## Project Structure

```
symmetry_discovery/
│
├── checkpoints/        # Saved model weights (downloaded)
├── figures/            # Generated visualizations
├── notebooks/
│   └── tasks.ipynb
├── scripts/
│   └── download_weights.py
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   └── __init__.py
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
pip install -r requirements.txt
```

### Usage

Run the notebook for full pipeline:

```bash
notebooks/tasks.ipynb
```

The notebook handles:

- model loading

* dataset generation
* training
* evaluation
* visualization

---

## Results & Visualizations

The project includes multiple evaluation tools:

- Latent space visualization
- Rotation trajectories
- Reconstruction quality
- Symmetry transformation paths

---

## Future Work

- Rotation-invariant networks (Bonus task)
- Extend to other transformations (scaling, translation)
- Apply to physics datasets
- Integrate group-theoretic constraints

---

## Author

Nirbhay Sharma

---

## License

This project is for research and educational purposes.

