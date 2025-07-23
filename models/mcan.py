"""MCAN (Modular Co-Attention Network) model (Yu et al., 2019)"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.base_model import BaseVQAModel


class SelfAttention(nn.Module):
    """Self-attention layer"""

    def __init__(self, hidden_dim: int, num_heads: int = 8):
        """
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear transformations
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim)
            mask: Mask (batch_size, seq_len)

        Returns:
            Output tensor
        """
        # Self-Attention
        residual = x
        x = self.norm1(x)

        # Multi-head attention
        batch_size, seq_len, _ = x.size()

        # Query, key, value computation
        q = (
            self.query(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Attention weight calculation
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        if mask is not None:
            mask = (
                mask.unsqueeze(1).unsqueeze(1).expand(-1, self.num_heads, seq_len, -1)
            )
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention
        attended = torch.matmul(attention_weights, v)
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_dim)
        )
        attended = self.output(attended)

        x = residual + attended

        # Feed Forward
        residual = x
        x = self.norm2(x)
        x = residual + self.feed_forward(x)

        return x


class GuidedAttention(nn.Module):
    """Guided attention layer"""

    def __init__(self, hidden_dim: int, num_heads: int = 8):
        """
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear transformations
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Query tensor (batch_size, seq_len_x, hidden_dim)
            y: Key-value tensor (batch_size, seq_len_y, hidden_dim)
            mask: Mask (batch_size, seq_len_x, seq_len_y)

        Returns:
            Output tensor
        """
        # Guided Attention
        residual = x
        x = self.norm1(x)
        y = self.norm1(y)

        # Multi-head attention
        batch_size, seq_len_x, _ = x.size()
        seq_len_y = y.size(1)

        # Query, key, value computation
        q = (
            self.query(x)
            .view(batch_size, seq_len_x, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(y)
            .view(batch_size, seq_len_y, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(y)
            .view(batch_size, seq_len_y, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Attention weight calculation
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention
        attended = torch.matmul(attention_weights, v)
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_x, self.hidden_dim)
        )
        attended = self.output(attended)

        x = residual + attended

        # Feed Forward
        residual = x
        x = self.norm2(x)
        x = residual + self.feed_forward(x)

        return x


class MCANEncoder(nn.Module):
    """MCAN encoder"""

    def __init__(self, hidden_dim: int, num_layers: int = 6, num_heads: int = 8):
        """
        Args:
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
            num_heads: Number of attention heads
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Question self-attention layers
        self.question_self_attn_layers = nn.ModuleList(
            [SelfAttention(hidden_dim, num_heads) for _ in range(num_layers)]
        )

        # Image self-attention layers
        self.image_self_attn_layers = nn.ModuleList(
            [SelfAttention(hidden_dim, num_heads) for _ in range(num_layers)]
        )

        # Guided attention layers (question → image)
        self.question_guided_attn_layers = nn.ModuleList(
            [GuidedAttention(hidden_dim, num_heads) for _ in range(num_layers)]
        )

        # Guided attention layers (image → question)
        self.image_guided_attn_layers = nn.ModuleList(
            [GuidedAttention(hidden_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(
        self, question_features: torch.Tensor, image_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            question_features: Question features (batch_size, seq_len, hidden_dim)
            image_features: Image features (batch_size, num_regions, hidden_dim)

        Returns:
            Encoded question features and image features
        """
        # Question self-attention
        for self_attn in self.question_self_attn_layers:
            question_features = self_attn(question_features)

        # Image self-attention
        for self_attn in self.image_self_attn_layers:
            image_features = self_attn(image_features)

        # Guided attention (question → image)
        for guided_attn in self.question_guided_attn_layers:
            question_features = guided_attn(question_features, image_features)

        # Guided attention (image → question)
        for guided_attn in self.image_guided_attn_layers:
            image_features = guided_attn(image_features, question_features)

        return question_features, image_features


class MCAN(BaseVQAModel):
    """MCAN (Modular Co-Attention Network) model"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        num_answers: int = 3129,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_layers: int = 6,
        num_heads: int = 8,
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
            num_layers: Number of MCAN layers
            num_heads: Number of attention heads
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

        self.num_layers = num_layers
        self.num_heads = num_heads

        # Use ResNet to extract image features
        resnet = models.resnet152(pretrained=pretrained)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-2])

        # Image feature projection
        self.image_projection = nn.Linear(
            2048, hidden_dim
        )  # ResNet152 output dimension

        # Question feature projection
        self.question_projection = nn.Linear(
            hidden_dim * 2, hidden_dim
        )  # Bidirectional LSTM output

        # MCAN encoder
        self.mcan_encoder = MCANEncoder(hidden_dim, num_layers, num_heads)

        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
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
        # Feature extraction with ResNet152
        features = self.image_encoder(images)  # (batch_size, 2048, H, W)

        # Treat spatial dimensions as regions
        batch_size, channels, height, width = features.size()
        features = features.view(batch_size, channels, -1)  # (batch_size, 2048, H*W)
        features = features.permute(0, 2, 1)  # (batch_size, H*W, 2048)

        # Projection
        features = self.image_projection(features)  # (batch_size, H*W, hidden_dim)

        return features

    def combine_features(
        self, image_features: torch.Tensor, question_features: torch.Tensor
    ) -> torch.Tensor:
        """Combine features (apply MCAN)"""
        # Project question features
        projected_question = self.question_projection(
            question_features
        )  # (batch_size, hidden_dim)

        # Expand question features (treat as sequence)
        question_seq = projected_question.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # MCAN encoder
        encoded_question, encoded_image = self.mcan_encoder(
            question_seq, image_features
        )

        # Global average pooling
        question_pooled = encoded_question.mean(dim=1)  # (batch_size, hidden_dim)
        image_pooled = encoded_image.mean(dim=1)  # (batch_size, hidden_dim)

        # Feature combination
        combined = torch.cat([question_pooled, image_pooled], dim=1)
        fused = self.feature_fusion(combined)

        return fused
