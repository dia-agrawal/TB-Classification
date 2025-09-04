TB-Classification
TB classification using AudioMAE Facebook
Unified Lightning Training Script
This repository provides a unified PyTorch Lightning training script that supports three modes:
Autoencoding (auto)
Binary Classification (classifier)
Triplet-loss Embedding (tripletloss)
It builds on Facebook’s AudioMAE implementation of Masked Autoencoders for audio, extending it with classification and triplet loss workflows.
Requirements
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
pyyaml>=5.4.0
tqdm>=4.62.0
pathlib
audio_mae module (wrapping Facebook’s AudioMAE)
Install dependencies:
pip install torch torchvision torchaudio pytorch-lightning torchinfo pyyaml
Reference
Our AudioMae model is referenced from the github link below and developed for our use. Facebook AudioMae: https://github.com/facebookresearch/AudioMAE/tree/main
