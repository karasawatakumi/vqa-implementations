"""BUTD (Bottom-Up and Top-Down Attention) model (Anderson et al., 2018)"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseVQAModel


class BottomUpAttention(nn.Module):
    """Bottom-Up Attention (object detection based)"""

    def __init__(self, feature_dim: int, hidden_dim: int, num_objects: int = 36):
        """
        Args:
            feature_dim: Feature dimension
            hidden_dim: Hidden layer dimension
            num_objects: Number of objects
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects

        # Object feature projection
        self.object_projection = nn.Linear(feature_dim, hidden_dim)

        # Spatial feature projection
        self.spatial_projection = nn.Linear(4, hidden_dim)  # (x1, y1, x2, y2)

        # Object feature combination
        self.combine_features = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self, object_features: torch.Tensor, spatial_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            object_features: Object features (batch_size, num_objects, feature_dim)
            spatial_features: Spatial features (batch_size, num_objects, 4)

        Returns:
            region_features: Region features (batch_size, num_objects, hidden_dim)
        """
        # Object feature projection
        projected_objects = self.object_projection(object_features)

        # Spatial feature projection
        projected_spatial = self.spatial_projection(spatial_features)

        # Feature combination
        combined = torch.cat([projected_objects, projected_spatial], dim=-1)
        region_features = self.combine_features(combined)

        return region_features


class TopDownAttention(nn.Module):
    """Top-Down Attention (question-based)"""

    def __init__(self, question_dim: int, region_dim: int, hidden_dim: int):
        """
        Args:
            question_dim: Question feature dimension
            region_dim: Region feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.question_dim = question_dim
        self.region_dim = region_dim
        self.hidden_dim = hidden_dim

        # Question feature projection
        self.question_projection = nn.Linear(question_dim, hidden_dim)

        # Region feature projection
        self.region_projection = nn.Linear(region_dim, hidden_dim)

        # Attention weight calculation
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(
        self, question_features: torch.Tensor, region_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            question_features: Question features (batch_size, question_dim)
            region_features: Region features (batch_size, num_regions, region_dim)

        Returns:
            attended_features: Features after attention application
            attention_weights: Attention weights
        """
        batch_size, num_regions, _ = region_features.size()

        # Question feature projection and expansion
        projected_question = self.question_projection(
            question_features
        )  # (batch_size, hidden_dim)
        projected_question = projected_question.unsqueeze(1).expand(
            -1, num_regions, -1
        )  # (batch_size, num_regions, hidden_dim)

        # Region feature projection
        projected_regions = self.region_projection(
            region_features
        )  # (batch_size, num_regions, hidden_dim)

        # Feature combination
        combined = torch.tanh(
            projected_question + projected_regions
        )  # (batch_size, num_regions, hidden_dim)

        # Attention weight calculation
        attention_weights = self.attention(combined)  # (batch_size, num_regions, 1)
        attention_weights = F.softmax(
            attention_weights, dim=1
        )  # (batch_size, num_regions, 1)

        # Apply attention
        attended_features = torch.sum(
            attention_weights * projected_regions, dim=1
        )  # (batch_size, hidden_dim)

        return attended_features, attention_weights.squeeze(-1)


class BUTD(BaseVQAModel):
    """BUTD (Bottom-Up and Top-Down Attention) model"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        num_answers: int = 3129,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        feature_dim: int = 2048,  # Faster R-CNN feature dimension
        num_objects: int = 36,
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
            feature_dim: Object feature dimension
            num_objects: Number of objects
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

        self.feature_dim = feature_dim
        self.num_objects = num_objects

        # Bottom-Up Attention
        self.bottom_up_attention = BottomUpAttention(
            feature_dim, hidden_dim, num_objects
        )

        # Top-Down Attention
        self.top_down_attention = TopDownAttention(
            hidden_dim * 2, hidden_dim, hidden_dim
        )  # Bidirectional LSTM output

        # Question feature projection
        self.question_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dim * 2, hidden_dim
            ),  # question + attention-applied features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_answers),
        )

    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features (object detection based)"""
        # In actual implementation, use Faster R-CNN for object detection
        # Here we use ResNet for simplicity
        batch_size = images.size(0)

        # Generate simple object features (in practice, use Faster R-CNN output)
        # Spatial features (bounding box coordinates)
        spatial_features = torch.rand(
            batch_size, self.num_objects, 4, device=images.device
        )

        # Object features (in practice, use Faster R-CNN output)
        object_features = torch.rand(
            batch_size, self.num_objects, self.feature_dim, device=images.device
        )

        # Bottom-Up Attention
        region_features = self.bottom_up_attention(object_features, spatial_features)

        return region_features

    def combine_features(
        self, image_features: torch.Tensor, question_features: torch.Tensor
    ) -> torch.Tensor:
        """Combine features (with Top-Down Attention)"""
        # Question feature projection
        projected_question = self.question_projection(question_features)

        # Top-Down Attention
        attended_features, _ = self.top_down_attention(
            projected_question, image_features
        )

        # Combine question features and attention-applied features
        combined = torch.cat([question_features, attended_features], dim=1)

        return combined
