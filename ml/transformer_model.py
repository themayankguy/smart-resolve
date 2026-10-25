"""
PyTorch Causal Transformer for Next-Action Prediction (Phase 5).

Predicts the player's next combat action (IDLE, MOVE, ATTACK, DODGE, HURT)
from a chronological sequence of preceding actions using causal multi-head self-attention.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocabulary mapping for action tokens
ACTION_VOCAB = {
    "PAD": 0,
    "IDLE": 1,
    "MOVE": 2,
    "ATTACK": 3,
    "DODGE": 4,
    "HURT": 5,
}

ACTION_NAMES = ["PAD", "IDLE", "MOVE", "ATTACK", "DODGE", "HURT"]
PLAYABLE_ACTIONS = ["IDLE", "MOVE", "ATTACK", "DODGE", "HURT"]
ACTION_TO_ID = ACTION_VOCAB
ID_TO_ACTION = {v: k for k, v in ACTION_VOCAB.items()}


class ActionTransformer(nn.Module):
    """
    Autoregressive Causal Transformer for sequential action forecasting.
    
    Architecture:
      - Token Embedding: vocab_size (6) -> d_model (32)
      - Learned Positional Embedding: max_seq_len (16) -> d_model (32)
      - Causal Multi-Head Self-Attention: 2 TransformerEncoderLayers, 4 heads
      - Classification Head: Linear(32, vocab_size) predicting next token logits
    """

    def __init__(
        self,
        vocab_size: int = len(ACTION_VOCAB),
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        max_seq_len: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_len = max_seq_len

        # Token & Positional Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=ACTION_VOCAB["PAD"])
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Causal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Next-Action Classification Head
        self.classifier = nn.Linear(d_model, vocab_size)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generates an upper-triangular causal attention mask to prevent attending to future tokens."""
        # Value of -inf prevents attention to future positions
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        return mask

    def forward(self, x: torch.Tensor, causal: bool = True, return_all_positions: bool = False) -> torch.Tensor:
        """
        Forward pass for action sequence input.
        
        Args:
            x: Tensor of shape (batch_size, seq_len) containing integer action token IDs.
            causal: If True, applies upper-triangular causal attention mask.
            return_all_positions: If True, returns logits for every position (batch, seq_len, vocab_size).
            
        Returns:
            Logits tensor of shape (batch_size, vocab_size) or (batch_size, seq_len, vocab_size).
        """
        b_sz, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(b_sz, seq_len)
        embeddings = self.token_embedding(x) + self.pos_embedding(positions)
        embeddings = self.layer_norm(self.dropout(embeddings))

        mask = self._generate_causal_mask(seq_len, x.device) if causal else None
        hidden = self.transformer(embeddings, mask=mask)

        if return_all_positions:
            return self.classifier(hidden)

        # Output logits predicted from the final sequence position
        last_hidden = hidden[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits

    def predict_next(self, tokens: list[int]) -> tuple[int, str, float, dict[str, float]]:
        """
        Infers the next action from a list of preceding action tokens.
        
        Returns:
            tuple (action_id, action_name, confidence, class_probabilities)
            where confidence is the maximum Softmax score (not a calibrated probability).
        """
        self.eval()
        # Truncate or pad to max_seq_len
        truncated = tokens[-self.max_seq_len:]
        if len(truncated) < self.max_seq_len:
            pad_len = self.max_seq_len - len(truncated)
            input_seq = [ACTION_VOCAB["PAD"]] * pad_len + truncated
        else:
            input_seq = truncated

        device = self.token_embedding.weight.device
        t_in = torch.tensor([input_seq], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = self.forward(t_in, causal=True)
            # Mask out PAD token from next-action predictions
            logits[0, ACTION_VOCAB["PAD"]] = -1e9
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

        best_id = int(np.argmax(probs))
        best_name = ACTION_NAMES[best_id]
        confidence = float(probs[best_id])

        prob_dict = {ACTION_NAMES[i]: float(round(float(probs[i]), 4)) for i in range(1, len(ACTION_NAMES))}
        return best_id, best_name, confidence, prob_dict
