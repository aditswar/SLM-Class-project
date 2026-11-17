# SLM Class Project: Small Language Model from Scratch

A small language model (SLM) built, trained, and used for text generation from scratch in PyTorch as a class project. The repository contains the full training notebook, a trained checkpoint, a custom tokenizer, and a notebook for generating stories with the trained model.

## Repository Structure

```
SLM-Class-project/
├── files/                    # Supporting files (data / assets)
├── slm_from_scratch.ipynb    # Model architecture, tokenizer training, and training loop
├── slm_from_scratch.pdf      # PDF export of the training notebook (for easy reading)
├── generateStory.ipynb       # Load the trained model and generate stories
├── slm_final.pt              # Trained model weights (final checkpoint)
├── tokenizer.json            # Trained tokenizer vocabulary and config
└── .gitattributes
```

## Overview

This project walks through the complete pipeline of building a language model end to end:

1. **Tokenization**: a tokenizer is trained on the dataset and saved to `tokenizer.json`.
2. **Model**: a small decoder-only Transformer language model is implemented from scratch in PyTorch.
3. **Training**: the model is trained with next-token prediction, and the final weights are saved to `slm_final.pt`.
4. **Generation**: `generateStory.ipynb` loads the checkpoint and tokenizer and samples new text from a prompt.

## Getting Started

### Prerequisites

- Python 3.9+
- [PyTorch](https://pytorch.org/)
- [Hugging Face `tokenizers`](https://github.com/huggingface/tokenizers)
- Jupyter Notebook or JupyterLab (or Google Colab)
- A GPU is recommended for training, but generation works on CPU

### Installation

```bash
git clone https://github.com/aditswar/SLM-Class-project.git
cd SLM-Class-project
pip install torch tokenizers jupyter numpy
```

## Usage

### Generate a story with the trained model

1. Open `generateStory.ipynb`.
2. Make sure `slm_final.pt` and `tokenizer.json` are in the same folder as the notebook.
3. Run all cells and edit the prompt to generate your own text.

```bash
jupyter notebook generateStory.ipynb
```

### Train or inspect the model

Open `slm_from_scratch.ipynb` to see the full implementation, including the model definition, data preparation, and training loop. If you just want to read it, `slm_from_scratch.pdf` has the same content without needing to run anything.

```bash
jupyter notebook slm_from_scratch.ipynb
```

## Files Explained

| File | Description |
|------|-------------|
| `slm_from_scratch.ipynb` | Full pipeline: data, tokenizer, model, training |
| `slm_from_scratch.pdf` | Static export of the training notebook |
| `generateStory.ipynb` | Inference notebook for text/story generation |
| `slm_final.pt` | Final trained PyTorch weights |
| `tokenizer.json` | Saved tokenizer used for encoding/decoding |
| `files/` | Supporting data and resources |

## Model Details

| Setting | Value |
|---------|-------|
| Architecture | Decoder-only Transformer |
| Parameters | _e.g. 1.3 M_ |
| Layers / Heads / Embedding dim | _e.g. 6 / 6 / 384_ |
| Context length | _e.g. 256_ |
| Vocabulary size | _e.g. 8,000_ |
| Training data | _dataset name_ |
| Hardware / Training time | _e.g. Colab T4, 1 hr_ |

## Acknowledgments

Built as a class project on small language models. Inspired by the GPT family of architectures and open educational resources on training language models from scratch.