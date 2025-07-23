"""SAN (Stacked Attention Networks) model (Yang et al., 2016)"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.base_model import BaseVQAModel


class AttentionLayer(nn.Module):
    """Attention layer"""

    def __init__(self, image_dim: int, question_dim: int, hidden_dim: int):
        """
        Args:
            image_dim: Image feature dimension
            question_dim: Question feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.image_dim = image_dim
        self.question_dim = question_dim
        self.hidden_dim = hidden_dim

        # Image feature projection
        self.image_projection = nn.Linear(image_dim, hidden_dim)

        # Question feature projection
        self.question_projection = nn.Linear(question_dim, hidden_dim)

        # Attention weight calculation
        self.attention = nn.Linear(hidden_dim, 1)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, image_features: torch.Tensor, question_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_features: Image features (batch_size, num_regions, image_dim)
            question_features: Question features (batch_size, question_dim)

        Returns:
            attended_features: Features after attention application
            attention_weights: Attention weights
        """
        batch_size, num_regions, _ = image_features.size()

        # Image feature projection
        projected_image = self.image_projection(
            image_features
        )  # (batch_size, num_regions, hidden_dim)

        # Question feature projection and expansion
        projected_question = self.question_projection(
            question_features
        )  # (batch_size, hidden_dim)
        projected_question = projected_question.unsqueeze(1).expand(
            -1, num_regions, -1
        )  # (batch_size, num_regions, hidden_dim)

        # Feature combination
        combined = torch.tanh(
            projected_image + projected_question
        )  # (batch_size, num_regions, hidden_dim)

        # Attention weight calculation
        attention_weights = self.attention(combined)  # (batch_size, num_regions, 1)
        attention_weights = F.softmax(
            attention_weights, dim=1
        )  # (batch_size, num_regions, 1)

        # Apply attention
        attended_features = torch.sum(
            attention_weights * projected_image, dim=1
        )  # (batch_size, hidden_dim)

        # Output projection
        output = self.output_projection(attended_features)  # (batch_size, hidden_dim)

        return output, attention_weights.squeeze(-1)


class SAN(BaseVQAModel):
    """SAN (Stacked Attention Networks) model"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        num_answers: int = 3129,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_attention_layers: int = 2,
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
            num_attention_layers: Number of attention layers
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

        self.num_attention_layers = num_attention_layers

        # Use VGG16 to extract image features
        vgg = models.vgg16(pretrained=pretrained)
        self.image_encoder = nn.Sequential(*list(vgg.features.children())[:-1])

        # Image feature dimension
        self.image_feature_dim = 512  # VGG16 final feature map channel count

        # Convert image features to region features
        self.region_projection = nn.Linear(self.image_feature_dim, hidden_dim)

        # Question feature projection
        self.question_projection = nn.Linear(
            hidden_dim * 2, hidden_dim
        )  # Bidirectional LSTM output

        # Attention layers
        self.attention_layers = nn.ModuleList(
            [
                AttentionLayer(hidden_dim, hidden_dim, hidden_dim)
                for _ in range(num_attention_layers)
            ]
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_answers),
        )

    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features"""
        # Feature extraction with VGG16
        features = self.image_encoder(images)  # (batch_size, 512, 14, 14)

        # Treat spatial dimensions as regions
        batch_size, channels, height, width = features.size()
        features = features.view(batch_size, channels, -1)  # (batch_size, 512, 196)
        features = features.permute(0, 2, 1)  # (batch_size, 196, 512)

        # Project to region features
        features = self.region_projection(features)  # (batch_size, 196, hidden_dim)

        return features

    def combine_features(
        self, image_features: torch.Tensor, question_features: torch.Tensor
    ) -> torch.Tensor:
        """Combine features (with attention)"""
        # Question feature projection
        projected_question = self.question_projection(
            question_features
        )  # (batch_size, hidden_dim)

        # Multi-layer attention
        current_features = image_features
        for attention_layer in self.attention_layers:
            current_features, _ = attention_layer(current_features, projected_question)

        return current_features
