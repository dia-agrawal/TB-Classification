# TB-Classification
TB classification using AudioMAE Facebook 

# Unified Lightning Training Script

This repository provides a **unified PyTorch Lightning** training script that supports three modes:

1. **Autoencoding** (`auto`)  
2. **Binary Classification** (`classifier`)  
3. **Triplet-loss Embedding** (`tripletloss`)  

It builds on Facebook’s AudioMAE implementation of Masked Autoencoders for audio, extending it with classification and triplet loss workflows.

## Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- PyTorch Lightning ≥ 1.7  
- torchaudio  
- torchinfo  
- `audio_mae` module (wrapping Facebook’s AudioMAE)  

Install dependencies:

pip install torch torchvision torchaudio pytorch-lightning torchinfo pyyaml


##**Reference** 
Our AudioMae model is referenced from the github link below and developed for our use. 
Facebook AudioMae: https://github.com/facebookresearch/AudioMAE/tree/main

