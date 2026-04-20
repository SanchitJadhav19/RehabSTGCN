"""
==============================================================================
  Model Inference Module for RehabSTGCN
==============================================================================

Clean wrapper around the trained RehabSTGCN model.
Handles model loading, device management, and inference.

Usage:
    predictor = RehabPredictor('checkpoints/best_model.pth')
    result = predictor.predict(skeleton_tensor)
    # result = {"score": 7.5, "attention": [0.01, 0.03, ...]}
==============================================================================
"""

import os
import sys
import torch
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from net.rehab_st_gcn import create_rehab_model


class RehabPredictor:
    """
    Wraps the trained RehabSTGCN model for easy inference.

    Usage:
        predictor = RehabPredictor('checkpoints/best_model.pth')
        result = predictor.predict(skeleton_tensor, length)
    """

    def __init__(self, model_path, device=None):
        """
        Args:
            model_path: Path to trained .pth checkpoint
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)

        # Create model architecture
        self.model = create_rehab_model(
            in_channels=3,
            graph_layout='openpose',
            graph_strategy='spatial'
        ).to(self.device)

        # Load trained weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle both full checkpoint and raw state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print(f"[RehabPredictor] Model loaded from {model_path} → {self.device}")

    @torch.no_grad()
    def predict(self, skeleton_tensor, length=None):
        """
        Run inference on a skeleton tensor.

        Args:
            skeleton_tensor: torch.Tensor of shape (1, 3, T, 18, 1)
            length: int or None — actual sequence length

        Returns:
            dict with:
                - score: float (predicted quality score)
                - attention: list of floats (attention weights per timestep)
        """
        # Move to device
        x = skeleton_tensor.to(self.device)

        # Prepare lengths tensor
        lengths = None
        if length is not None:
            lengths = torch.tensor([length], dtype=torch.long).to(self.device)

        # Run inference
        score, attention = self.model(x, lengths)

        return {
            'score': round(float(score.squeeze().cpu().item()), 2),
            'attention': attention.squeeze().cpu().numpy().tolist(),
        }


# ============================================================================
# Quick test
# ============================================================================
if __name__ == '__main__':
    print("Testing RehabPredictor...")

    model_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'best_model.pth')

    if not os.path.exists(model_path):
        print(f"No model found at {model_path}. Run training first.")
    else:
        predictor = RehabPredictor(model_path)

        # Dummy input
        dummy = torch.randn(1, 3, 300, 18, 1)
        result = predictor.predict(dummy, length=200)

        print(f"✓ Score: {result['score']}")
        print(f"✓ Attention length: {len(result['attention'])}")
        print(f"✓ Top-3 attended timesteps: "
              f"{sorted(range(len(result['attention'])), key=lambda i: result['attention'][i], reverse=True)[:3]}")
