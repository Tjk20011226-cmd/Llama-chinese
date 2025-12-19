
# Llama-Chinese (MiniPoet)

This repository provides the supplementary materials for the paper:

**Chinese Poetry Generation Using a Lightweight Model with Attention-Assisted Loss Function**

## Overview
This repository contains the training configurations, evaluation scripts, and experimental results
used in the above paper.  
It is intended to support reproducibility and facilitate further research on lightweight poetry generation models.

## Repository Structure
- `configs/`: Training and inference configuration files
- `scripts/`: Training, inference scripts
- `evaluation/`: Evaluation metrics and Manual Evalution sample
- `ablation/`: Ablation study 
- `results/`: Experimental results and generated samples

## Model Weights
Due to data licensing constraints and deployment considerations, the trained model weights are not publicly released.
However, all training configurations and evaluation scripts are provided.
Model weights can be made available for academic research purposes upon reasonable request.

## Data
The datasets used in this study are publicly available.
We provide detailed descriptions of data sources and preprocessing procedures in the paper.
Please refer to `data/README.md` for instructions on obtaining and preparing the datasets.

## Baseline Models
All baseline models (e.g., Qwen2-7B, ChatGLM3-6B) were evaluated using their publicly released checkpoints without additional fine-tuning on poetry-specific data.
The exact model versions and evaluation settings are documented in the paper to ensure fair comparison.

## Reproducibility
All experiments were conducted under unified decoding settings.
The released configuration files and scripts allow reproduction of the reported experimental results.
