"""VQA dataset class"""

import json
import os
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.vocab import Vocabulary

# Type definitions for VQA data
VQADataItem = Dict[str, Any]
VQABatch = Dict[str, Any]


class VQADataset(Dataset):
    """VQA dataset class"""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        vocab: Optional[Vocabulary] = None,
        max_question_length: int = 20,
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            data_dir: Data directory
            split: Data split (train, val, test)
            vocab: Vocabulary object
            max_question_length: Maximum question length
            image_size: Image size
            transform: Image transformation
        """
        self.data_dir = data_dir
        self.split = split
        self.vocab = vocab
        self.max_question_length = max_question_length
        self.image_size = image_size

        # Image transformation
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform

        # Load data
        self.data = self._load_data()

        # Build vocabulary (training only)
        if vocab is None and split == "train":
            self.vocab = self._build_vocab()

    def _load_data(self) -> List[Dict]:
        """Load data"""
        # Question file
        question_file = os.path.join(
            self.data_dir, f"v2_Questions_{self.split.capitalize()}_mscoco.json"
        )

        # Annotation file (not needed for test)
        annotation_file = None
        if self.split != "test":
            annotation_file = os.path.join(
                self.data_dir, f"v2_Annotations_{self.split.capitalize()}_mscoco.json"
            )

        # Load question data
        with open(question_file, "r") as f:
            questions = json.load(f)

        # Load annotation data
        annotations = None
        if annotation_file and os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                annotations = json.load(f)

        # Organize data
        data = []
        questions_dict = {q["question_id"]: q for q in questions["questions"]}

        if annotations:
            for ann in annotations["annotations"]:
                question_id = ann["question_id"]
                question = questions_dict[question_id]

                # Image path
                image_id = question["image_id"]
                image_path = os.path.join(
                    self.data_dir,
                    f"{self.split}2014",
                    f"COCO_{self.split}2014_{image_id:012d}.jpg",
                )

                # Answers
                answers = [ans["answer"] for ans in ann["answers"]]

                data.append(
                    {
                        "question_id": question_id,
                        "image_id": image_id,
                        "image_path": image_path,
                        "question": question["question"],
                        "answers": answers,
                        "question_type": ann.get("question_type", ""),
                        "answer_type": ann.get("answer_type", ""),
                    }
                )
        else:
            # For test data
            for question in questions["questions"]:
                image_id = question["image_id"]
                image_path = os.path.join(
                    self.data_dir,
                    f"{self.split}2014",
                    f"COCO_{self.split}2014_{image_id:012d}.jpg",
                )

                data.append(
                    {
                        "question_id": question["question_id"],
                        "image_id": image_id,
                        "image_path": image_path,
                        "question": question["question"],
                        "answers": [],
                        "question_type": "",
                        "answer_type": "",
                    }
                )

        return data

    def _build_vocab(self) -> Vocabulary:
        """Build vocabulary"""
        vocab = Vocabulary(min_freq=3, max_size=10000)

        # Build vocabulary from questions
        questions = [item["question"] for item in self.data]
        vocab.build_from_texts(questions)

        return vocab

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> VQADataItem:
        """Get data item"""
        item = self.data[idx]

        # Load and transform image
        image = Image.open(item["image_path"]).convert("RGB")
        image = self.transform(image)

        # Encode question
        if self.vocab is None:
            raise ValueError("Vocabulary is not initialized")

        question_indices = self.vocab.encode(item["question"])

        # Padding
        if len(question_indices) < self.max_question_length:
            question_indices += [self.vocab.word2idx["<PAD>"]] * (
                self.max_question_length - len(question_indices)
            )
        else:
            question_indices = question_indices[: self.max_question_length]

        question_tensor = torch.tensor(question_indices, dtype=torch.long)

        # Question length
        question_length = torch.tensor(
            min(len(self.vocab.encode(item["question"])), self.max_question_length),
            dtype=torch.long,
        )

        result = {
            "image": image,
            "question": question_tensor,
            "question_length": question_length,
            "question_id": item["question_id"],
            "image_id": item["image_id"],
        }

        # Include answers during training
        if self.split != "test" and item["answers"]:
            result["answers"] = item["answers"]
            result["question_type"] = item["question_type"]
            result["answer_type"] = item["answer_type"]

        return result


class VQACollateFn:
    """VQA data collation function"""

    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    def __call__(self, batch: List[VQADataItem]) -> VQABatch:
        """Organize batch data"""
        # Images
        images = torch.stack([item["image"] for item in batch])

        # Questions
        questions = torch.stack([item["question"] for item in batch])
        question_lengths = torch.stack([item["question_length"] for item in batch])

        # Metadata
        question_ids = [item["question_id"] for item in batch]
        image_ids = [item["image_id"] for item in batch]

        result = {
            "images": images,
            "questions": questions,
            "question_lengths": question_lengths,
            "question_ids": question_ids,
            "image_ids": image_ids,
        }

        # Include answers during training
        if "answers" in batch[0]:
            answers = [item["answers"] for item in batch]
            question_types = [item["question_type"] for item in batch]
            answer_types = [item["answer_type"] for item in batch]

            result.update(
                {
                    "answers": answers,
                    "question_types": question_types,
                    "answer_types": answer_types,
                }
            )

        return result
