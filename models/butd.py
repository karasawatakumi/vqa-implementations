"""BUTD (Bottom-Up and Top-Down Attention) model (Anderson et al., 2018)"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import faster_rcnn_resnet50_fpn

from models.base_model import BaseVQAModel


class FasterRCNNFeatureExtractor(nn.Module):
    """Faster R-CNN feature extractor for object detection"""

    def __init__(self, pretrained: bool = True, num_objects: int = 36):
        """
        Args:
            pretrained: Whether to use pretrained model
            num_objects: Number of objects to extract
        """
        super().__init__()

        # Load pretrained Faster R-CNN
        self.faster_rcnn = faster_rcnn_resnet50_fpn(pretrained=pretrained)

        # Feature dimension from ResNet backbone
        self.feature_dim = 2048
        self.num_objects = num_objects

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract object features using Faster R-CNN

        Args:
            images: Input images (batch_size, 3, H, W)

        Returns:
            object_features: Object features (batch_size, num_objects, feature_dim)
            spatial_features: Spatial features (batch_size, num_objects, 4)
        """
        batch_size = images.size(0)

        # Run Faster R-CNN
        with torch.no_grad():
            detections = self.faster_rcnn(images)

        # Process detections
        object_features_list = []
        spatial_features_list = []

        for i in range(batch_size):
            boxes = detections[i]["boxes"]  # (num_detections, 4)
            scores = detections[i]["scores"]  # (num_detections,)

            # Filter by confidence threshold
            keep = scores > 0.5
            boxes = boxes[keep]
            scores = scores[keep]

            # Limit number of objects
            if len(boxes) > self.num_objects:
                # Keep top scoring objects
                _, indices = torch.topk(scores, self.num_objects)
                boxes = boxes[indices]
                scores = scores[indices]

            # Extract features for detected objects
            if len(boxes) > 0:
                # Use ROI pooling to extract features
                roi_features = self._extract_roi_features(images[i : i + 1], boxes)

                # Pad or truncate to fixed number of objects
                if len(boxes) < self.num_objects:
                    # Pad with zeros
                    padding_size = self.num_objects - len(boxes)
                    roi_features = F.pad(roi_features, (0, 0, 0, padding_size))
                    boxes = F.pad(boxes, (0, 0, 0, padding_size))
                else:
                    # Truncate
                    roi_features = roi_features[: self.num_objects]
                    boxes = boxes[: self.num_objects]
            else:
                # No detections, use zeros
                roi_features = torch.zeros(
                    self.num_objects, self.feature_dim, device=images.device
                )
                boxes = torch.zeros(self.num_objects, 4, device=images.device)

            object_features_list.append(roi_features)
            spatial_features_list.append(boxes)

        # Stack batch
        object_features = torch.stack(object_features_list, dim=0)
        spatial_features = torch.stack(spatial_features_list, dim=0)

        return object_features, spatial_features

    def _extract_roi_features(
        self, image: torch.Tensor, boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract ROI features using ROI pooling

        Args:
            image: Single image (1, 3, H, W)
            boxes: Bounding boxes (num_objects, 4)

        Returns:
            roi_features: ROI features (num_objects, feature_dim)
        """
        # Use ROI pooling from Faster R-CNN backbone
        features = self.faster_rcnn.backbone(image)

        # Use ROI pooling to extract features for each box
        roi_features = []
        for box in boxes:
            roi_feat = self._extract_region_feature(features, box)
            roi_features.append(roi_feat)

        return torch.stack(roi_features, dim=0)

    def _extract_region_feature(
        self, features: dict, box: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract feature for a specific region

        Args:
            features: Feature maps from backbone
            box: Bounding box (4,)

        Returns:
            region_feature: Region feature (feature_dim,)
        """
        # Use the last feature map (highest resolution)
        feature_map = features["3"]  # (1, 2048, H, W)

        # Convert box coordinates to feature map coordinates
        h, w = feature_map.shape[2:]
        x1, y1, x2, y2 = box

        # Scale coordinates to feature map size (assuming input size 800)
        x1 = max(0, min(int(x1 * w / 800), w - 1))
        y1 = max(0, min(int(y1 * h / 800), h - 1))
        x2 = max(x1 + 1, min(int(x2 * w / 800), w))
        y2 = max(y1 + 1, min(int(y2 * h / 800), h))

        # Extract region
        region = feature_map[0, :, y1:y2, x1:x2]

        # Global average pooling
        region_feature = F.adaptive_avg_pool2d(region, (1, 1)).squeeze()

        return region_feature


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
        expanded_question = projected_question.unsqueeze(1).expand(
            batch_size, num_regions, self.hidden_dim
        )  # (batch_size, num_regions, hidden_dim)

        # Region feature projection
        projected_regions = self.region_projection(
            region_features
        )  # (batch_size, num_regions, hidden_dim)

        # Attention calculation
        combined = (
            expanded_question + projected_regions
        )  # (batch_size, num_regions, hidden_dim)
        attention_weights = torch.softmax(
            self.attention(combined).squeeze(-1), dim=1
        )  # (batch_size, num_regions)

        # Apply attention
        attended_features = torch.sum(
            region_features * attention_weights.unsqueeze(-1), dim=1
        )  # (batch_size, region_dim)

        return attended_features, attention_weights


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

        # Faster R-CNN feature extractor
        self.feature_extractor = FasterRCNNFeatureExtractor(
            pretrained=pretrained, num_objects=num_objects
        )

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
        """Extract image features using Faster R-CNN object detection"""
        # Extract object features using Faster R-CNN
        object_features, spatial_features = self.feature_extractor(images)

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
