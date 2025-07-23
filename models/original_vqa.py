"""Original VQA model (Antol et al., 2015)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.base_model import BaseVQAModel


class OriginalVQA(BaseVQAModel):
    """Original VQA model"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        num_answers: int = 3129,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
    ):
        """
        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            num_answers: Number of answer candidates
            learning_rate: Learning rate
            weight_decay: Weight decay
            pretrained: Whether to use pretrained model
        """
        super().__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_answers=num_answers,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        # Use VGG16 to extract image features
        vgg = models.vgg16(pretrained=pretrained)
        self.image_encoder = nn.Sequential(*list(vgg.features.children())[:-1])

        # Get image feature dimension
        self.image_feature_dim = 512 * 14 * 14  # VGG16 final feature map size

        # Layer to compress image features
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Update classifier (combine image and question features)
        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dim * 3, hidden_dim
            ),  # image + question (bidirectional LSTM)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_answers),
        )

    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features"""
        # Feature extraction with VGG16
        features = self.image_encoder(images)  # (batch_size, 512, 14, 14)

        # Global average pooling
        features = F.adaptive_avg_pool2d(features, (1, 1))  # (batch_size, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size, 512)

        # Projection
        features = self.image_projection(features)  # (batch_size, hidden_dim)

        return features

    def combine_features(
        self, image_features: torch.Tensor, question_features: torch.Tensor
    ) -> torch.Tensor:
        """Combine features"""
        # Simple concatenation
        combined = torch.cat([image_features, question_features], dim=1)
        return combined
