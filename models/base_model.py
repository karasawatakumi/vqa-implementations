"""Base class for VQA models"""

from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy

# Type definitions for VQA data
VQABatch = Dict[str, Any]


class BaseVQAModel(pl.LightningModule):
    """Base class for VQA models"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        num_answers: int = 3129,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        """
        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            num_answers: Number of answer candidates
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        super().__init__()
        self.save_hyperparameters()

        # Parameters
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_answers = num_answers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Question encoder
        self.question_encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Embedding layer
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_answers),
        )

        # Evaluation metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_answers)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_answers)

    def encode_question(
        self, question: torch.Tensor, question_length: torch.Tensor
    ) -> torch.Tensor:
        """Encode question"""
        # Embedding
        embedded = self.word_embedding(question)  # (batch_size, seq_len, embed_dim)

        # Encode with LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, question_length.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.question_encoder(packed)

        # Concatenate final hidden states
        hidden = torch.cat(
            [hidden[0], hidden[1]], dim=1
        )  # (batch_size, hidden_dim * 2)

        return hidden

    def forward(
        self,
        images: torch.Tensor,
        questions: torch.Tensor,
        question_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass"""
        # Extract image features (implemented in subclasses)
        image_features = self.extract_image_features(images)

        # Encode question
        question_features = self.encode_question(questions, question_lengths)

        # Combine features
        combined_features = self.combine_features(image_features, question_features)

        # Classification
        logits = self.classifier(combined_features)

        return logits

    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features (implemented in subclasses)"""
        raise NotImplementedError

    def combine_features(
        self, image_features: torch.Tensor, question_features: torch.Tensor
    ) -> torch.Tensor:
        """Combine features (implemented in subclasses)"""
        raise NotImplementedError

    def training_step(self, batch: VQABatch, batch_idx: int) -> torch.Tensor:
        """Training step"""
        images = batch["images"]
        questions = batch["questions"]
        question_lengths = batch["question_lengths"]
        answers = batch["answers"]

        # Convert answers to labels (use most frequent answer)
        labels = self._get_answer_labels(answers)

        # Forward pass
        logits = self(images, questions, question_lengths)

        # Calculate loss
        loss = F.cross_entropy(logits, labels)

        # Calculate accuracy
        acc = self.train_acc(logits, labels)

        # Log
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch: VQABatch, batch_idx: int) -> torch.Tensor:
        """Validation step"""
        images = batch["images"]
        questions = batch["questions"]
        question_lengths = batch["question_lengths"]
        answers = batch["answers"]

        # Convert answers to labels
        labels = self._get_answer_labels(answers)

        # Forward pass
        logits = self(images, questions, question_lengths)

        # Calculate loss
        loss = F.cross_entropy(logits, labels)

        # Calculate accuracy
        acc = self.val_acc(logits, labels)

        # Log
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def _get_answer_labels(self, answers: List[List[str]]) -> torch.Tensor:
        """Convert answers to labels"""
        # Select most frequent answer
        labels = []
        for answer_list in answers:
            # Count answer occurrences
            answer_counts: Dict[str, int] = {}
            for answer in answer_list:
                answer_counts[answer] = answer_counts.get(answer, 0) + 1

            # Select most frequent answer
            most_common = max(answer_counts.items(), key=lambda x: x[1])[0]
            labels.append(self.answer_to_label(most_common))

        return torch.tensor(labels, device=str(self.device))

    def answer_to_label(self, answer: str) -> int:
        """Convert answer to label (simple implementation)"""
        # In actual implementation, use a predefined answer dictionary
        # Here we use hash for simplicity
        return hash(answer) % self.num_answers

    def configure_optimizers(self):
        """Configure optimizers"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
