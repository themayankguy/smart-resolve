"""
PyTorch Autoencoder Architecture for Combat Anomaly Detection (Phase 4).

Reconstructs 11-dimensional combat feature windows. Windows exhibiting
substantially elevated reconstruction error relative to the training distribution
are flagged as anomalous combat patterns.

CRITICAL ATML NOTE:
This module performs statistical anomaly detection (identifying atypical combat distributions).
It does NOT perform cheating or malicious intent detection, and must not be described as such.
"""

import torch
import torch.nn as nn


class CombatAutoencoder(nn.Module):
    """
    Symmetric modular autoencoder mapping 11D combat windows into a 4D latent bottleneck.
    
    Architecture:
      Encoder: 11 -> 8 -> 4 (Latent Space)
      Decoder: 4 -> 8 -> 11 (Reconstruction)
    """

    def __init__(self, in_features: int = 11, latent_dim: int = 4, hidden_dim: int = 8):
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder branch: Compresses 11D feature space into 4D latent representation
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.1),
        )

        # Decoder branch: Reconstructs 11D feature space from 4D latent bottleneck
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, in_features),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes normalized feature vectors into latent representations."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent vectors back into the original 11D feature space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder and decoder.
        
        Returns:
            tuple (reconstructed_x, latent_z)
        """
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z

    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes per-sample Mean Squared Error (MSE) reconstruction loss:
            error_i = (1 / D) * sum_{j=1}^D (x_{i,j} - x_hat_{i,j})^2
            
        Returns:
            torch.Tensor of shape (B,) containing non-negative MSE per sample.
        """
        self.eval()
        with torch.no_grad():
            x_rec, _ = self.forward(x)
            mse_per_sample = torch.mean((x - x_rec) ** 2, dim=-1)
        return mse_per_sample
