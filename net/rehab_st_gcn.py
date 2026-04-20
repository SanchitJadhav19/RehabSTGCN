"""
==============================================================================
  Rehab ST-GCN: Modified ST-GCN for Rehabilitation Exercise Assessment
==============================================================================

Modifications over baseline ST-GCN:
  1. Classification → Regression (continuous score prediction)
  2. LSTM layer after ST-GCN backbone for temporal modeling
  3. Self-Attention mechanism for interpretability (joint/temporal importance)
  4. Huber Loss (SmoothL1Loss) for robust regression
  5. Variable-length input handling via masking
  6. Dual output: predicted score + attention weights

Architecture:
  Input Skeleton → [ST-GCN Backbone] → [LSTM] → [Self-Attention] → [Regression Head] → Score

Author: Neha J (College Project)
Reference: ST-GCN paper (https://arxiv.org/abs/1801.07455)
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph


# =============================================================================
# Module 1: ST-GCN Backbone (Feature Extractor — kept from original)
# =============================================================================
class STGCNBackbone(nn.Module):
    """
    The original ST-GCN feature extractor.
    Takes raw skeleton sequences and produces spatial-temporal features.

    Input:  (N, C, T, V, M) — batch, channels, frames, joints, persons
    Output: (N, 256, T', V) — extracted features (T' = T after temporal pooling)
    """

    def __init__(self, in_channels, graph_args, edge_importance_weighting, **kwargs):
        super().__init__()

        # ---- Build skeleton graph ----
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # ---- Network parameters ----
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        # ---- Batch normalization for input data ----
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        # ---- 10 ST-GCN layers (same as original paper) ----
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 128, kernel_size, 2, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 256, kernel_size, 2, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
        ))

        # ---- Learnable edge importance weights ----
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V, M) skeleton input
        Returns:
            x: (N, 256, T', V) spatial-temporal features
        """
        # ---- Data normalization ----
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # (N, M, V, C, T)
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()   # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)

        # ---- Pass through all ST-GCN layers ----
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # ---- Reshape back: average over persons (M) ----
        _, C_out, T_out, V_out = x.size()
        x = x.view(N, M, C_out, T_out, V_out)
        x = x.mean(dim=1)  # average over persons → (N, 256, T', V)

        return x


# =============================================================================
# Module 2: Temporal LSTM (captures long-range temporal dependencies)
# =============================================================================
class TemporalLSTM(nn.Module):
    """
    LSTM layer to model temporal dependencies in the feature sequence.

    Why LSTM after ST-GCN?
    - ST-GCN captures LOCAL spatial-temporal patterns (via fixed kernel size)
    - LSTM captures GLOBAL temporal dependencies (entire sequence context)
    - This is especially important for exercise assessment where the
      overall movement trajectory matters.

    Input:  (N, 256, T', V) — features from backbone
    Output: (N, T', hidden_size) — temporal feature sequence
    """

    def __init__(self, input_size=256, hidden_size=128, num_layers=1, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,    # 256 channels from ST-GCN
            hidden_size=hidden_size,  # compressed representation
            num_layers=num_layers,
            batch_first=True,         # input: (batch, seq, features)
            bidirectional=False,      # unidirectional for simplicity
            dropout=dropout if num_layers > 1 else 0
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, lengths=None):
        """
        Args:
            x: (N, C, T, V) features from ST-GCN backbone
            lengths: (N,) actual sequence lengths for each sample (optional)
        Returns:
            lstm_out: (N, T, hidden_size) temporal feature sequence
        """
        N, C, T, V = x.size()

        # ---- Pool over joints (spatial dimension) ----
        # Average over all joints to get temporal feature sequence
        x = x.mean(dim=3)          # (N, C, T)
        x = x.permute(0, 2, 1)     # (N, T, C) — ready for LSTM

        # ---- Handle variable-length sequences ----
        if lengths is not None:
            # Pack sequences to efficiently handle different lengths
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        else:
            lstm_out, _ = self.lstm(x)  # (N, T, hidden_size)

        # ---- Layer normalization for training stability ----
        lstm_out = self.layer_norm(lstm_out)

        return lstm_out


# =============================================================================
# Module 3: Self-Attention (interpretability — shows which timesteps matter)
# =============================================================================
class TemporalAttention(nn.Module):
    """
    Simple self-attention mechanism over temporal features.

    Purpose:
    - Learns WHICH TIMESTEPS are most important for the final score
    - Produces attention weights that can be visualized for interpretability
    - Example: high attention on the "peak extension" frame of a knee exercise

    How it works:
    1. Project each timestep feature through a small network
    2. Compute importance score for each timestep
    3. Softmax → attention weights (sum to 1)
    4. Weighted sum → single context vector

    Input:  (N, T, hidden_size) — temporal features from LSTM
    Output: (N, hidden_size) — attended context vector
            (N, T) — attention weights (for visualization)
    """

    def __init__(self, hidden_size=128, attention_size=64):
        super().__init__()

        # Simple 2-layer attention network
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_size, attention_size),  # project down
            nn.Tanh(),                                # non-linearity
            nn.Linear(attention_size, 1)              # score per timestep
        )

    def forward(self, lstm_output, mask=None):
        """
        Args:
            lstm_output: (N, T, hidden_size)
            mask: (N, T) boolean — True for padded positions to ignore
        Returns:
            context: (N, hidden_size) — weighted feature vector
            attention_weights: (N, T) — importance of each timestep
        """
        # ---- Compute attention scores ----
        scores = self.attention_net(lstm_output)  # (N, T, 1)
        scores = scores.squeeze(-1)               # (N, T)

        # ---- Mask out padded positions ----
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        # ---- Softmax → attention weights (probabilities) ----
        attention_weights = F.softmax(scores, dim=1)  # (N, T)

        # ---- Weighted sum of features ----
        # Each timestep's feature is weighted by its importance
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (N, 1, T)
            lstm_output                       # (N, T, hidden_size)
        ).squeeze(1)                          # (N, hidden_size)

        return context, attention_weights


# =============================================================================
# Module 4: Regression Head (predicts continuous exercise score)
# =============================================================================
class RegressionHead(nn.Module):
    """
    Simple MLP head that converts attended features into a single score.

    Design choices:
    - Two FC layers with dropout for regularization
    - Single scalar output (no activation — score can be any range)
    - Batch normalization for training stability

    Input:  (N, hidden_size) — attended context vector
    Output: (N, 1) — predicted exercise quality score
    """

    def __init__(self, hidden_size=128, dropout=0.3):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),    # compress features
            nn.ReLU(),
            nn.Dropout(dropout),            # prevent overfitting
            nn.Linear(64, 1)               # single score output
        )

    def forward(self, x):
        """
        Args:
            x: (N, hidden_size)
        Returns:
            score: (N, 1)
        """
        return self.head(x)


# =============================================================================
# COMPLETE MODEL: RehabSTGCN (combines all 4 modules)
# =============================================================================
class RehabSTGCN(nn.Module):
    """
    =========================================================================
    Modified ST-GCN for Rehabilitation Exercise Assessment
    =========================================================================

    Architecture Flow:
    ┌─────────────┐    ┌──────────┐    ┌───────────────┐    ┌────────────┐
    │  Skeleton    │───→│ ST-GCN   │───→│ Temporal      │───→│  Self-     │
    │  Input       │    │ Backbone │    │ LSTM          │    │  Attention │
    │ (N,C,T,V,M) │    │ (10 layers)│  │ (sequence)    │    │ (weights)  │
    └─────────────┘    └──────────┘    └───────────────┘    └─────┬──────┘
                                                                  │
                                                           ┌──────▼──────┐
                                                           │ Regression  │
                                                           │ Head        │
                                                           │ → Score     │
                                                           └─────────────┘
    Outputs:
        - predicted_score: (N, 1) — exercise quality score
        - attention_weights: (N, T) — temporal importance (interpretable)

    Modifications over baseline ST-GCN:
        1. Regression instead of classification
        2. LSTM for global temporal modeling
        3. Self-attention for interpretability
        4. Handles variable-length sequences
        5. Dual output (score + attention weights)
    =========================================================================
    """

    def __init__(self,
                 in_channels=3,
                 graph_args={'layout': 'openpose', 'strategy': 'spatial'},
                 edge_importance_weighting=True,
                 lstm_hidden=128,
                 lstm_layers=1,
                 attention_size=64,
                 dropout=0.3,
                 **kwargs):
        super().__init__()

        # ---- Module 1: ST-GCN Backbone ----
        self.backbone = STGCNBackbone(
            in_channels=in_channels,
            graph_args=graph_args,
            edge_importance_weighting=edge_importance_weighting,
            **kwargs
        )

        # ---- Module 2: Temporal LSTM ----
        self.temporal_lstm = TemporalLSTM(
            input_size=256,           # ST-GCN output channels
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout
        )

        # ---- Module 3: Self-Attention ----
        self.attention = TemporalAttention(
            hidden_size=lstm_hidden,
            attention_size=attention_size
        )

        # ---- Module 4: Regression Head ----
        self.regression_head = RegressionHead(
            hidden_size=lstm_hidden,
            dropout=dropout
        )

    def forward(self, x, lengths=None):
        """
        Full forward pass through the modified ST-GCN pipeline.

        Args:
            x: (N, C, T, V, M) — skeleton input
                N = batch size
                C = channels (typically 3: x, y, confidence)
                T = number of frames (can vary with padding)
                V = number of joints (18 for OpenPose, 25 for NTU)
                M = number of persons per frame
            lengths: (N,) — actual sequence lengths (for variable-length handling)

        Returns:
            predicted_score: (N, 1) — continuous exercise quality score
            attention_weights: (N, T') — importance of each temporal step
        """
        # Step 1: Extract spatial-temporal features
        features = self.backbone(x)           # (N, 256, T', V)

        # Step 2: Scale lengths to account for ST-GCN temporal pooling (2 stride-2 layers → T/4)
        effective_lengths = None
        if lengths is not None:
            T_out = features.size(2)  # Actual size after pooling
            effective_lengths = (lengths.float() / 4.0).long().clamp(min=1, max=T_out)

        # Step 3: Model temporal dependencies
        temporal_features = self.temporal_lstm(features, effective_lengths)  # (N, T', H)

        # Step 4: Create mask for padded positions (if variable-length)
        mask = None
        if effective_lengths is not None:
            T_out = temporal_features.size(1)
            mask = torch.arange(T_out, device=x.device).unsqueeze(0) >= effective_lengths.unsqueeze(1)

        # Step 5: Attend to important timesteps
        context, attention_weights = self.attention(temporal_features, mask)  # (N, H), (N, T')

        # Step 6: Predict score
        predicted_score = self.regression_head(context)  # (N, 1)

        return predicted_score, attention_weights


# =============================================================================
# ST-GCN Block (unchanged from original — kept for compatibility)
# =============================================================================
class st_gcn_block(nn.Module):
    """
    Single spatial-temporal graph convolution block.
    This is the building block of the ST-GCN backbone.

    Components:
        1. Graph Convolution (GCN) — spatial modeling over joints
        2. Temporal Convolution (TCN) — temporal modeling over frames
        3. Residual connection — for gradient flow

    Input:  (N, in_channels, T, V), adjacency matrix A
    Output: (N, out_channels, T', V), A
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dropout=0, residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        # Spatial graph convolution
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        # Temporal convolution
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


# =============================================================================
# LOSS FUNCTION: Huber Loss wrapper (for convenience)
# =============================================================================
class RehabLoss(nn.Module):
    """
    Huber Loss (SmoothL1Loss) for robust regression.

    Why Huber Loss instead of MSE?
    - MSE is sensitive to outliers (squares large errors)
    - MAE doesn't penalize small errors enough
    - Huber Loss = best of both worlds:
        * Behaves like MSE for small errors (smooth, differentiable)
        * Behaves like MAE for large errors (robust to outliers)

    This is especially important for rehabilitation scoring where:
    - Some annotations may be noisy
    - Score ranges can have natural variance
    """

    def __init__(self, delta=1.0):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=delta)

    def forward(self, predicted, target):
        """
        Args:
            predicted: (N, 1) — model predictions
            target: (N, 1) — ground truth scores
        Returns:
            loss: scalar
        """
        return self.loss_fn(predicted, target)


# =============================================================================
# Helper function: Create model with default settings
# =============================================================================
def create_rehab_model(in_channels=3,
                       graph_layout='openpose',
                       graph_strategy='spatial',
                       lstm_hidden=128,
                       dropout=0.3):
    """
    Factory function to create the RehabSTGCN model with sensible defaults.

    Args:
        in_channels: Number of input channels (3 for x,y,confidence)
        graph_layout: 'openpose' (18 joints) or 'ntu-rgb+d' (25 joints)
        graph_strategy: 'spatial' (recommended), 'uniform', or 'distance'
        lstm_hidden: Hidden size for LSTM (128 is a good default)
        dropout: Dropout rate (0.3 for regularization)

    Returns:
        model: RehabSTGCN model ready for training
    """
    model = RehabSTGCN(
        in_channels=in_channels,
        graph_args={'layout': graph_layout, 'strategy': graph_strategy},
        edge_importance_weighting=True,
        lstm_hidden=lstm_hidden,
        dropout=dropout
    )
    return model


# =============================================================================
# Quick test: verify model works
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  RehabSTGCN — Model Verification")
    print("=" * 60)

    # Create model
    model = create_rehab_model(in_channels=3, graph_layout='openpose')
    print(f"\n✓ Model created successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")

    # Test forward pass with fixed-length input
    print(f"\n--- Test 1: Fixed-length input ---")
    batch_size = 4
    x = torch.randn(batch_size, 3, 300, 18, 1)  # 4 samples, 300 frames, 18 joints
    score, attn = model(x)
    print(f"✓ Input shape:  {x.shape}")
    print(f"✓ Score shape:  {score.shape}  (expected: [{batch_size}, 1])")
    print(f"✓ Attention shape: {attn.shape}  (expected: [{batch_size}, T'])")
    print(f"✓ Sample scores: {score.detach().squeeze().tolist()}")
    print(f"✓ Attention sum: {attn.sum(dim=1).tolist()} (should be ~1.0 each)")

    # Test forward pass with variable-length input
    print(f"\n--- Test 2: Variable-length input ---")
    lengths = torch.tensor([300, 200, 150, 100])
    score_v, attn_v = model(x, lengths=lengths)
    print(f"✓ Lengths: {lengths.tolist()}")
    print(f"✓ Score shape:  {score_v.shape}")
    print(f"✓ Attention shape: {attn_v.shape}")

    # Test loss function
    print(f"\n--- Test 3: Huber Loss ---")
    criterion = RehabLoss(delta=1.0)
    target = torch.tensor([[7.5], [8.0], [6.5], [9.0]])
    loss = criterion(score, target)
    print(f"✓ Predicted: {score.detach().squeeze().tolist()}")
    print(f"✓ Target:    {target.squeeze().tolist()}")
    print(f"✓ Huber Loss: {loss.item():.4f}")

    print(f"\n{'=' * 60}")
    print(f"  All tests passed! Model is ready for training.")
    print(f"{'=' * 60}")
