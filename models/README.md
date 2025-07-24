# VQA Models Implementation

This directory contains implementations of four major approaches for Visual Question Answering (VQA).

## 📁 File Structure

- `base_model.py` - Base class for all VQA models
- `original_vqa.py` - Original VQA (Antol et al., 2015)
- `san.py` - Stacked Attention Networks (Yang et al., 2016)
- `butd.py` - Bottom-Up and Top-Down Attention (Anderson et al., 2018)
- `mcan.py` - Modular Co-Attention Network (Yu et al., 2019)

## 🏗️ Architecture Design

### BaseVQAModel (Base Class)

All VQA models inherit from the `BaseVQAModel` class and are implemented using PyTorch Lightning.

#### Core Components

1. **Question Encoder**: Bidirectional LSTM for question encoding
2. **Word Embedding**: Word embedding layer
3. **Classifier**: Final answer prediction classifier
4. **Evaluation Metrics**: Training and validation accuracy calculation

#### Abstract Methods

- `extract_image_features()`: Image feature extraction (implemented in subclasses)
- `combine_features()`: Image and question feature combination (implemented in subclasses)

## 🤖 Implemented Models

### 1. Original VQA (Antol et al., 2015)

**File**: `original_vqa.py`

**Features**:
- VGG16-based image feature extraction
- LSTM-based question encoding
- Simple concatenation and MLP for answer generation
- Basic VQA architecture

**Implementation Details**:
```python
class OriginalVQA(BaseVQAModel):
    def extract_image_features(self, images):
        # VGG16 feature extraction
        # Global average pooling
        # Linear layer for dimension adjustment
    
    def combine_features(self, image_features, question_features):
        # Simple concatenation
        # MLP for classification
```

### 2. SAN - Stacked Attention Networks (Yang et al., 2016)

**File**: `san.py`

**Features**:
- Multi-layer attention mechanism
- Question-guided focus on relevant image regions
- Progressive attention refinement
- Fine-grained visual reasoning

**Implementation Details**:
```python
class SAN(BaseVQAModel):
    def __init__(self, num_attention_layers=2):
        # Multiple attention layers
        # Question-image attention
    
    def extract_image_features(self, images):
        # CNN feature extraction
        # Spatial feature map generation
    
    def combine_features(self, image_features, question_features):
        # Multi-layer attention
        # Progressive feature refinement
```

### 3. BUTD - Bottom-Up and Top-Down Attention (Anderson et al., 2018)

**File**: `butd.py`

**Features**:
- Faster R-CNN-based object detection feature extraction
- Bottom-up: Object-level feature extraction
- Top-down: Question-guided attention
- Structured visual representation

**Implementation Details**:
```python
class BUTD(BaseVQAModel):
    def __init__(self, feature_dim=2048, num_objects=36):
        # Object detection features
        # Spatial attention
    
    def extract_image_features(self, images):
        # Faster R-CNN features
        # Object region features
    
    def combine_features(self, image_features, question_features):
        # Spatial attention
        # Question-object attention
```

### 4. MCAN - Modular Co-Attention Network (Yu et al., 2019)

**File**: `mcan.py`

**Features**:
- Modular attention mechanism design
- Cross-modal attention between image and text
- Combination of multiple attention modules
- Learning complex visual-linguistic interactions

**Implementation Details**:
```python
class MCAN(BaseVQAModel):
    def __init__(self, num_layers=6, num_heads=8):
        # Modular attention
        # Multi-head attention
    
    def extract_image_features(self, images):
        # CNN feature extraction
        # Spatial feature maps
    
    def combine_features(self, image_features, question_features):
        # Modular attention
        # Cross-modal interactions
```

## 🔧 Usage

### Model Initialization

```python
from models.original_vqa import OriginalVQA
from models.san import SAN
from models.butd import BUTD
from models.mcan import MCAN

# Configuration
config = {
    "model": {
        "name": "original_vqa",
        "vocab_size": 10000,
        "embed_dim": 512,
        "hidden_dim": 1024,
        "num_answers": 3129,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "pretrained": True
    }
}

# Model creation
model = OriginalVQA(
    vocab_size=config["model"]["vocab_size"],
    embed_dim=config["model"]["embed_dim"],
    hidden_dim=config["model"]["hidden_dim"],
    num_answers=config["model"]["num_answers"],
    learning_rate=config["model"]["learning_rate"],
    weight_decay=config["model"]["weight_decay"],
    pretrained=config["model"]["pretrained"]
)
```

### Inference

```python
# Input data
images = torch.randn(1, 3, 224, 224)  # Batch size 1 image
questions = torch.randint(0, 1000, (1, 20))  # Question tokens
question_lengths = torch.tensor([15])  # Question length

# Inference
with torch.no_grad():
    logits = model(images, questions, question_lengths)
    probs = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)
```

## 📊 Expected Performance

Expected performance on VQA v2.0 validation set:

| Model | Accuracy | VQA Score |
|-------|----------|-----------|
| Original VQA | ~50% | ~0.45 |
| SAN | ~55% | ~0.50 |
| BUTD | ~65% | ~0.60 |
| MCAN | ~70% | ~0.65 |

## 🔄 Extensibility

To add a new VQA model:

1. Inherit from `BaseVQAModel`
2. Implement `extract_image_features()` and `combine_features()`
3. Define additional components as needed
4. Add the new model to configuration files

```python
class NewVQAModel(BaseVQAModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # New components
    
    def extract_image_features(self, images):
        # New image feature extraction
    
    def combine_features(self, image_features, question_features):
        # New feature combination
```

## 📝 Notes

- All models use PyTorch Lightning
- Configuration is managed via YAML files
- Pre-trained models can be used
- Automatic metrics calculation and logging 