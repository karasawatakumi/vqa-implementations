# VQA Implementation

A project implementing four major approaches for Visual Question Answering (VQA). Uses PyTorch Lightning to provide a consistent framework from training to evaluation and inference.

## 📋 Implemented Approaches

### 1. Original VQA (Antol et al., 2015)
- **Paper**: [VQA: Visual Question Answering](https://arxiv.org/abs/1505.00468)
- **Features**: 
  - Image feature extraction using VGG16
  - Question encoding using LSTM
  - Simple concatenation and MLP for answer generation
  - Basic VQA architecture

### 2. SAN - Stacked Attention Networks (Yang et al., 2016)
- **Paper**: [Stacked Attention Networks for Image Question Answering](https://arxiv.org/abs/1511.02274)
- **Features**:
  - Multi-layer attention mechanism
  - Question-guided focus on relevant image regions
  - Progressive attention refinement
  - Fine-grained visual reasoning

### 3. BUTD - Bottom-Up and Top-Down Attention (Anderson et al., 2018)
- **Paper**: [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)
- **Features**:
  - Object detection-based feature extraction using Faster R-CNN
  - Bottom-up: Object-level feature extraction
  - Top-down: Question-guided attention
  - Structured visual representation

### 4. MCAN - Modular Co-Attention Network (Yu et al., 2019)
- **Paper**: [Deep Modular Co-Attention Networks for Visual Question Answering](https://arxiv.org/abs/1906.10770)
- **Features**:
  - Modular attention mechanism design
  - Cross-modal attention between image and text
  - Combination of multiple attention modules
  - Learning complex visual-linguistic interactions

## 🚀 Setup

### 1. Environment Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone git@github.com:karasawatakumi/vqa-implementations.git
cd vqa

# Create virtual environment and install dependencies
# uv will automatically download and manage the required Python version (3.11)
uv sync
```

### 2. Dataset Preparation

#### Download VQA v2.0 Dataset

Download the following files from the [VQA official website](https://visualqa.org/download.html) and place them in the `data/vqa_v2/` directory:

**Image Data:**
- `train2014.zip` (COCO train images, ~13GB)
- `val2014.zip` (COCO val images, ~6GB)

**Question Data:**
- `v2_Questions_Train_mscoco.zip`
- `v2_Questions_Val_mscoco.zip`

**Annotation Data:**
- `v2_Annotations_Train_mscoco.zip`
- `v2_Annotations_Val_mscoco.zip`

#### Data Preprocessing

```bash
# Extract zip files and preprocess dataset
python scripts/prepare_vqa_dataset.py --extract_zips

# Download pretrained models (optional)
python scripts/download_pretrained_models.py
```

The preprocessing script will:
- Extract all zip files automatically
- Build vocabulary from questions
- Create answer vocabulary
- Generate processed dataset files
- Save vocabulary to `data/vocab.pkl`
- Create processed data files in `data/processed/`

### 3. Training

```bash
# Original VQA
python tools/train.py --config configs/original_vqa.yaml

# SAN
python tools/train.py --config configs/san.yaml

# BUTD
python tools/train.py --config configs/butd.yaml

# MCAN
python tools/train.py --config configs/mcan.yaml
```


### 4. Evaluation

```bash
# Evaluate on validation data
python tools/evaluate.py \
    --model_path checkpoints/original_vqa/best_model.ckpt \
    --config configs/original_vqa.yaml \
    --split val
```

### 5. Inference

```bash
# Inference on single image
python tools/inference.py \
    --model_path checkpoints/original_vqa/best_model.ckpt \
    --config configs/original_vqa.yaml \
    --image_path path/to/image.jpg \
    --question "What is in this image?"
```

## 📁 Project Structure

```
vqa/
├── pyproject.toml
├── README.md
├── .gitignore                  # Git ignore settings
├── configs/                    # Configuration files
├── models/                     # Model implementations
├── datasets/                   # Dataset classes
├── utils/                      # Utility functions
├── scripts/                    # Scripts
├── tools/                      # Training, evaluation, and inference tools
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── inference.py           # Inference script
├── data/                      # Data directory
├── checkpoints/               # Trained models
└── logs/                      # Log files
```

## ⚙️ Configuration

Each model's configuration is managed in YAML files in the `configs/` directory.

### Main Configuration Items

```yaml
model:
  name: "original_vqa"          # Model name
  vocab_size: 10000            # Vocabulary size
  embed_dim: 512               # Embedding dimension
  hidden_dim: 1024             # Hidden layer dimension
  num_answers: 3129            # Number of answer candidates
  learning_rate: 1e-3          # Learning rate
  weight_decay: 1e-4           # Weight decay

data:
  data_dir: "data/vqa_v2"      # Data directory
  max_question_length: 20      # Maximum question length
  image_size: 224              # Image size
  batch_size: 32               # Batch size
  num_workers: 4               # DataLoader workers

training:
  max_epochs: 50               # Maximum epochs
  val_check_interval: 0.25     # Validation interval
  accumulate_grad_batches: 1   # Gradient accumulation
  gradient_clip_val: 0.5       # Gradient clipping
  early_stopping_patience: 5   # Early stopping patience
```

## 📊 Evaluation Metrics

The following evaluation metrics are implemented:

- **Accuracy**: Accuracy against 10 annotators' answers
- **VQA Score**: Score based on how many out of 10 annotators gave the same answer
- **Question Type Accuracy**: Accuracy by question type
- **Answer Type Accuracy**: Accuracy by answer type

## 📈 Performance

Expected performance for each model (VQA v2.0 validation set):

| Model | Accuracy | VQA Score |
|-------|----------|-----------|
| Original VQA | ~50% | ~0.45 |
| SAN | ~55% | ~0.50 |
| BUTD | ~65% | ~0.60 |
| MCAN | ~70% | ~0.65 |

## 🛠️ Development Setup

This project uses the following tools for code quality:

- **Black**: Code formatter
- **isort**: Import sorter  
- **flake8**: Linter
- **mypy**: Type checker

### Pre-commit

```bash
# install pre-commit
pip install pre-commit

# install pre-commit hooks
pre-commit install

# check all files
pre-commit run --all-files
```
