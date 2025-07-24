# VQA Dataset Implementation

This directory contains implementations for processing and loading Visual Question Answering (VQA) datasets.

## 📁 File Structure

- `vqa_dataset.py` - VQA dataset class and collation function
- `__init__.py` - Package initialization file

## 🗂️ Dataset Overview

### VQA v2.0 Dataset

This project uses the VQA v2.0 dataset.

**Data Composition**:
- **Images**: COCO dataset images (train2014, val2014)
- **Questions**: Natural language questions
- **Answers**: 10 annotator answers (training and validation only)

**Data Splits**:
- **Training**: 443,757 questions (82,783 images)
- **Validation**: 214,354 questions (40,504 images)
- **Test**: 447,793 questions (81,434 images)

## 🏗️ Implementation Details

### VQADataset Class

A VQA dataset class that inherits from PyTorch's `Dataset` class.

#### Core Features

1. **Data Loading**: Question and annotation data loading from JSON files
2. **Image Preprocessing**: Resize, normalization, tensor conversion
3. **Vocabulary Building**: Vocabulary creation during training
4. **Batch Processing**: Collation for efficient batch processing

#### Initialization Parameters

```python
VQADataset(
    data_dir: str,                    # Data directory
    split: str = "train",             # Data split (train/val/test)
    vocab: Optional[Vocabulary] = None, # Vocabulary object
    max_question_length: int = 20,    # Maximum question length
    image_size: int = 224,            # Image size
    transform: Optional[transforms.Compose] = None  # Custom transform
)
```

#### Data Structure

Each data item has the following structure:

```python
{
    "image_path": str,           # Image file path
    "question": str,             # Question text
    "question_id": int,          # Question ID
    "image_id": int,             # Image ID
    "answers": List[str],        # Answer list (training/validation only)
    "answer_type": str,          # Answer type (training/validation only)
    "question_type": str         # Question type (training/validation only)
}
```

### VQACollateFn Class

Collation function for batch processing.

#### Features

1. **Padding**: Unify question lengths
2. **Tensor Conversion**: Convert data to tensor format
3. **Batch Creation**: Structure data for efficient batch processing

#### Output Format

```python
{
    "images": torch.Tensor,           # (batch_size, 3, H, W)
    "questions": torch.Tensor,        # (batch_size, max_length)
    "question_lengths": torch.Tensor, # (batch_size,)
    "answers": List[List[str]],       # Answer lists (training/validation only)
    "question_ids": List[int],        # Question IDs
    "image_ids": List[int]            # Image IDs
}
```

## 🔧 Usage

### Basic Usage Example

```python
from datasets.vqa_dataset import VQADataset, VQACollateFn
from utils.vocab import Vocabulary

# Load vocabulary (or create new)
vocab = Vocabulary.load("data/vocab.pkl")

# Create datasets
train_dataset = VQADataset(
    data_dir="data/vqa_v2",
    split="train",
    vocab=vocab,
    max_question_length=20,
    image_size=224
)

val_dataset = VQADataset(
    data_dir="data/vqa_v2",
    split="val",
    vocab=vocab,
    max_question_length=20,
    image_size=224
)

# Create data loaders
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=VQACollateFn(vocab),
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    collate_fn=VQACollateFn(vocab),
    pin_memory=True
)
```

### Data Inspection

```python
# Dataset sizes
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Sample data inspection
sample = train_dataset[0]
print(f"Image path: {sample['image_path']}")
print(f"Question: {sample['question']}")
print(f"Answers: {sample['answers']}")
print(f"Question type: {sample['question_type']}")
print(f"Answer type: {sample['answer_type']}")

# Batch data inspection
batch = next(iter(train_loader))
print(f"Images shape: {batch['images'].shape}")
print(f"Questions shape: {batch['questions'].shape}")
print(f"Question lengths: {batch['question_lengths']}")
```

## 📊 Data Statistics

### Question Type Distribution

- **Yes/No**: ~45%
- **Number**: ~10%
- **Other**: ~45%

### Answer Type Distribution

- **Yes/No**: ~45%
- **Number**: ~10%
- **Other**: ~45%

### Vocabulary Statistics

- **Vocabulary size**: ~10,000 words (configurable)
- **Average question length**: ~6 words
- **Maximum question length**: 20 words (configurable)

## 🔄 Customization

### Adding Custom Transforms

```python
from torchvision import transforms

# Define custom transform
custom_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Dataset with custom transform
dataset = VQADataset(
    data_dir="data/vqa_v2",
    split="train",
    vocab=vocab,
    transform=custom_transform
)
```

### Adding New Datasets

To add a new VQA dataset:

1. Inherit from `VQADataset` class
2. Override `_load_data()` method
3. Customize `_build_vocab()` method if needed

```python
class CustomVQADataset(VQADataset):
    def _load_data(self) -> List[Dict]:
        # Custom data loading process
        pass
    
    def _build_vocab(self) -> Vocabulary:
        # Custom vocabulary building process
        pass
```

## 📝 Notes

### Memory Usage

- Images are loaded on-demand (memory efficient)
- Vocabulary is pre-built and saved
- Batch size should be adjusted based on GPU memory

### Performance

- Set `num_workers` appropriately (based on CPU cores)
- Use `pin_memory=True` for faster GPU transfer
- Pre-process data when possible

### Data Integrity

- Verify image file existence
- Validate question-answer correspondence
- Check vocabulary consistency

## 🔍 Troubleshooting

### Common Issues

1. **Image files not found**
   - Check data directory path
   - Verify image file existence

2. **Vocabulary errors**
   - Check vocabulary file loading
   - Verify vocabulary size settings

3. **Memory issues**
   - Reduce batch size
   - Reduce `num_workers`

### Debugging Methods

```python
# Detailed dataset inspection
dataset = VQADataset(data_dir="data/vqa_v2", split="train")
print(f"Dataset size: {len(dataset)}")
print(f"Sample data: {dataset[0]}")

# Vocabulary inspection
print(f"Vocabulary size: {dataset.vocab.vocab_size}")
print(f"Word to index: {dataset.vocab.word2idx}")
``` 