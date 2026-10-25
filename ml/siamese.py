"""
PyTorch Siamese Network architecture and Contrastive Loss for Combat Playstyle Similarity.
Implements shared-weight metric learning to map combat feature windows into a metric embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNet(nn.Module):
    """
    Siamese Network with shared encoder weights for playstyle similarity learning.
    
    Architecture:
        Input: [Batch, InFeatures=11]
        -> Linear(11 -> 32) -> BatchNorm1d(32) -> ReLU()
        -> Linear(32 -> 16) -> BatchNorm1d(16) -> ReLU()
        -> Linear(16 -> EmbeddingDim=8)
        Output Embedding: [Batch, EmbeddingDim=8]
    """

    def __init__(self, in_features: int = 11, embedding_dim: int = 8):
        super().__init__()
        self.in_features = in_features
        self.embedding_dim = embedding_dim

        # Shared feedforward encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, embedding_dim),
        )

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes a single feature vector into the metric embedding space.
        Input shape: [Batch, InFeatures] -> Output shape: [Batch, EmbeddingDim]
        """
        # Handle single sample without batch dimension if necessary
        if x.dim() == 1:
            x = x.unsqueeze(0)
            return self.encoder(x).squeeze(0)
        return self.encoder(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Simultaneously passes pair (x1, x2) through the same shared encoder.
        Output: (embedding1, embedding2)
        """
        z1 = self.forward_one(x1)
        z2 = self.forward_one(x2)
        return z1, z2

    def compute_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Computes Euclidean distance between embeddings of x1 and x2."""
        z1, z2 = self.forward(x1, x2)
        return F.pairwise_distance(z1, z2)


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for metric learning (Hadsell et al., 2006):
    
        L(z1, z2, y) = y * D^2 + (1 - y) * max(0, margin - D)^2
        
    where:
        - D = ||z1 - z2||_2 (Euclidean distance)
        - y = 1.0 for positive/similar pairs (penalizes distance)
        - y = 0.0 for negative/dissimilar pairs (penalizes distances smaller than margin)
        - margin: minimum target separation distance between dissimilar styles
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        euclidean_dist = F.pairwise_distance(z1, z2)
        loss_similar = label * torch.pow(euclidean_dist, 2)
        loss_dissimilar = (1.0 - label) * torch.pow(
            torch.clamp(self.margin - euclidean_dist, min=0.0), 2
        )
        return torch.mean(loss_similar + loss_dissimilar)
