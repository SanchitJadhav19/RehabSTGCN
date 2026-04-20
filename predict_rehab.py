"""
==============================================================================
  Prediction Script for RehabSTGCN
==============================================================================

Loads a trained model and predicts the quality score for a single skeleton
sequence. Also visualizes the temporal attention weights.

Usage:
  python predict_rehab.py --model_path checkpoints/best_model.pth
==============================================================================
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net.rehab_st_gcn import create_rehab_model
from feeder.feeder_rehab import RehabFeeder


def parse_args():
    parser = argparse.ArgumentParser(description='RehabSTGCN Prediction')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                        help='Path to trained model (.pth)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to skeleton data (.npy)')
    parser.add_argument('--score_path', type=str, default=None,
                        help='Path to true scores (.npy) - optional')
    parser.add_argument('--index', type=int, default=0,
                        help='Index of the sample to predict')
    parser.add_argument('--save_plot', type=str, default='attention_plot.png',
                        help='Path to save attention plot')
    return parser.parse_args()


def predict():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n--- Loading Model: {args.model_path} ---")
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Create and load model
    model = create_rehab_model(in_channels=3, graph_layout='openpose').to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded successfully")

    # Load data
    if args.data_path is None:
        # Use synthetic data if no path provided
        print("--- Using synthetic data for prediction demo ---")
        from feeder.feeder_rehab import create_synthetic_dataset
        data_path, score_path = create_synthetic_dataset(num_samples=10)
    else:
        data_path = args.data_path
        score_path = args.score_path

    dataset = RehabFeeder(data_path, score_path, window_size=300)
    data, true_score, length = dataset[args.index]
    
    # Prepare input
    # data: (C, T, V, M) -> (1, C, T, V, M)
    input_data = data.unsqueeze(0).to(device)
    input_len = length.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        pred_score, attn_weights = model(input_data, input_len)

    print(f"\n--- Prediction Results (Sample {args.index}) ---")
    print(f"  Ground Truth Score: {true_score.item():.2f}")
    print(f"  Predicted Score:    {pred_score.item():.2f}")
    print(f"  Error:              {abs(true_score.item() - pred_score.item()):.2f}")

    # Visualize Attention
    attn = attn_weights[0].cpu().numpy()
    
    # Text-based visualization
    print("\n--- Temporal Attention (Importance) ---")
    max_attn = attn.max()
    for i in range(0, len(attn), 5):  # show every 5th frame for brevity
        bar = "█" * int(attn[i] / max_attn * 30)
        print(f"  Frame {i:3d}: {bar} {attn[i]:.4f}")

    # Plot and save
    plt.figure(figsize=(10, 4))
    plt.plot(attn, color='teal', linewidth=2)
    plt.fill_between(range(len(attn)), attn, color='teal', alpha=0.3)
    plt.title(f'Temporal Attention Weights (Predicted Score: {pred_score.item():.2f})')
    plt.xlabel('Temporal Step (T\')')
    plt.ylabel('Importance Weight')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(args.save_plot)
    print(f"\n✓ Attention plot saved to: {args.save_plot}")


if __name__ == '__main__':
    predict()
