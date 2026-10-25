"""
Graph Convolutional Network (GCN) for Dungeon Room Threat Prediction.
Implements spectral graph convolution in plain PyTorch (Kipf & Welling, ICLR 2017)
without external graph library dependencies.

Mathematical Formulation:
    H^(l+1) = activation( D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2) * H^(l) * W^(l) )
where:
    A_tilde = A + I_N       (Adjacency matrix with added self-loops)
    D_tilde_ii = sum_j(A_tilde_ij)  (Degree matrix with self-loops)
    D_tilde^(-1/2)          (Inverse square-root degree matrix)
    H^(0) = X in R^(N x F)  (Input node feature matrix for N rooms)
    W^(l) in R^(F_l x F_out)(Learnable weight tensor)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

THREAT_CLASSES = ["LOW", "MEDIUM", "HIGH"]
THREAT_TO_ID = {name: idx for idx, name in enumerate(THREAT_CLASSES)}
ID_TO_THREAT = {idx: name for idx, name in enumerate(THREAT_CLASSES)}


def build_normalized_adjacency(num_nodes: int, edges: list[tuple[int, int]], device: torch.device | None = None) -> torch.Tensor:
    """
    Constructs the symmetrically normalized adjacency matrix with self-loops:
        A_hat = D_tilde^(-1/2) * (A + I_N) * D_tilde^(-1/2)

    Handles isolated nodes gracefully: nodes with degree 0 in A obtain degree 1
    via the self-loop (I_N), avoiding division by zero.

    Args:
        num_nodes: Number of rooms N in the dungeon graph.
        edges: List of undirected corridor connections (u, v).
        device: Target torch device.

    Returns:
        Tensor of shape (num_nodes, num_nodes) representing A_hat.
    """
    device = device or torch.device("cpu")
    if num_nodes <= 0:
        return torch.zeros((0, 0), dtype=torch.float32, device=device)

    # 1. Base Adjacency Matrix A (undirected, symmetric)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    for u, v in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u, v] = 1.0
            adj[v, u] = 1.0

    # 2. Add Self-Loops: A_tilde = A + I_N
    adj_tilde = adj + torch.eye(num_nodes, dtype=torch.float32, device=device)

    # 3. Compute Degree Matrix D_tilde
    degree = torch.sum(adj_tilde, dim=1)  # Shape: (N,)

    # 4. Inverse Square Root Degree: D_tilde^(-1/2)
    # Since self-loops are added, degree[i] >= 1.0 for all nodes, preventing zero division.
    deg_inv_sqrt = torch.pow(degree, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # 5. Symmetric Normalization: D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2)
    d_mat = torch.diag(deg_inv_sqrt)
    norm_adj = torch.matmul(torch.matmul(d_mat, adj_tilde), d_mat)

    return norm_adj


class GCNLayer(nn.Module):
    """
    Single Graph Convolutional Layer.
    Computes:
        Z = A_hat * (X * W) + b
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier uniform initialization for stable gradient flow."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GCN layer.

        Args:
            x: Node feature matrix of shape (N, in_features).
            norm_adj: Normalized adjacency matrix A_hat of shape (N, N).

        Returns:
            Output node features of shape (N, out_features).
        """
        # Linear feature transformation: (N, in_features) x (in_features, out_features) -> (N, out_features)
        support = torch.matmul(x, self.weight)

        # Neighborhood message aggregation: (N, N) x (N, out_features) -> (N, out_features)
        output = torch.matmul(norm_adj, support)

        if self.bias is not None:
            output = output + self.bias
        return output


class RoomThreatGCN(nn.Module):
    """
    2-Layer Graph Convolutional Network with room-level classification head.
    Predicts threat categories (LOW, MEDIUM, HIGH) for every room in the dungeon.
    """

    def __init__(self, in_features: int = 7, hidden_dim: int = 32, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Layer 1: Feature expansion & 1-hop neighborhood aggregation
        self.gcn1 = GCNLayer(in_features, hidden_dim)
        # Layer 2: 2-hop neighborhood aggregation
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(p=dropout)
        # Final room-level linear classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node feature matrix of shape (N, in_features)
            norm_adj: Symmetrically normalized adjacency matrix (N, N)

        Returns:
            Logits of shape (N, num_classes) for each room.
        """
        # Block 1
        h1 = self.gcn1(x, norm_adj)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)

        # Block 2
        h2 = self.gcn2(h1, norm_adj)
        h2 = F.relu(h2)
        h2 = self.dropout(h2)

        # Classification logits
        logits = self.classifier(h2)
        return logits


class NonGraphMLP(nn.Module):
    """
    Ablation Baseline: Multi-Layer Perceptron using the exact same node features
    and layer widths as RoomThreatGCN, but without graph adjacency convolutions.
    Establishes the value of topological graph structure beyond isolated node features.
    """

    def __init__(self, in_features: int = 7, hidden_dim: int = 32, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes logits without message passing."""
        return self.net(x)
