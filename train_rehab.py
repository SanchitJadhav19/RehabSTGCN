"""
==============================================================================
  Training Script for RehabSTGCN
==============================================================================

Complete training pipeline with:
  - Huber Loss (SmoothL1Loss) for robust regression
  - Learning rate scheduling (ReduceLROnPlateau)
  - Early stopping to prevent overfitting
  - Training/validation split
  - Attention weight visualization
  - Model checkpointing

Usage:
  python train_rehab.py                          # synthetic data demo
  python train_rehab.py --data_path data.npy \
                        --score_path scores.npy  # real data
==============================================================================
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net.rehab_st_gcn import RehabSTGCN, RehabLoss, create_rehab_model
from feeder.feeder_rehab import RehabFeeder, rehab_collate_fn, create_synthetic_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='RehabSTGCN Training')

    # Data
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to skeleton data (.npy)')
    parser.add_argument('--score_path', type=str, default=None,
                        help='Path to scores file (.npy or .pkl)')
    parser.add_argument('--window_size', type=int, default=300,
                        help='Sequence length (frames)')
    parser.add_argument('--num_joints', type=int, default=18,
                        help='Number of skeleton joints')

    # Model
    parser.add_argument('--graph_layout', type=str, default='openpose',
                        choices=['openpose', 'ntu-rgb+d'])
    parser.add_argument('--graph_strategy', type=str, default='spatial',
                        choices=['spatial', 'uniform', 'distance'])
    parser.add_argument('--lstm_hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')

    # Output
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='Use synthetic data for demo')

    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    num_batches = 0

    for data, scores, lengths in loader:
        data = data.to(device)       # (N, C, T, V, M)
        scores = scores.to(device)   # (N, 1)
        lengths = lengths.to(device)  # (N,)

        # Forward pass
        predicted, attention = model(data, lengths)

        # Compute loss
        loss = criterion(predicted, scores)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevents exploding gradients with LSTM)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation set. Returns loss, MAE, and sample attention."""
    model.eval()
    total_loss = 0
    total_mae = 0
    num_batches = 0
    sample_attention = None

    for data, scores, lengths in loader:
        data = data.to(device)
        scores = scores.to(device)
        lengths = lengths.to(device)

        predicted, attention = model(data, lengths)
        loss = criterion(predicted, scores)

        # Mean Absolute Error
        mae = torch.abs(predicted - scores).mean().item()

        total_loss += loss.item()
        total_mae += mae
        num_batches += 1

        # Save one attention map for visualization
        if sample_attention is None:
            sample_attention = attention[0].cpu().numpy()

    avg_loss = total_loss / max(num_batches, 1)
    avg_mae = total_mae / max(num_batches, 1)

    return avg_loss, avg_mae, sample_attention


def visualize_attention(attention_weights, save_path=None):
    """
    Simple text-based visualization of attention weights.
    Shows which timesteps the model focuses on.
    """
    print("\n  Attention Weights (temporal importance):")
    print("  " + "─" * 50)

    # Normalize for display
    weights = attention_weights
    max_w = weights.max()
    num_bars = min(len(weights), 40)  # show at most 40 timesteps
    step = max(1, len(weights) // num_bars)

    for i in range(0, len(weights), step):
        bar_len = int(weights[i] / max_w * 30)
        bar = "█" * bar_len
        print(f"  t={i:3d} │{bar} {weights[i]:.4f}")

    print("  " + "─" * 50)

    # Optional: save attention to file
    if save_path:
        np.save(save_path, attention_weights)
        print(f"  → Saved attention to {save_path}")


def main():
    args = parse_args()

    # ---- Device setup ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  RehabSTGCN Training Pipeline")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    # ---- Prepare data ----
    if args.data_path is None or args.use_synthetic:
        print(f"\n  → Using SYNTHETIC data for demo...")
        data_path, score_path = create_synthetic_dataset(
            num_samples=80,
            num_frames=args.window_size,
            num_joints=args.num_joints
        )
    else:
        data_path = args.data_path
        score_path = args.score_path

    # ---- Create dataset ----
    dataset = RehabFeeder(
        data_path=data_path,
        score_path=score_path,
        window_size=args.window_size
    )

    # ---- Train/val split (80/20) ----
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=rehab_collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=rehab_collate_fn,
        num_workers=0
    )

    print(f"  Train samples: {train_size}")
    print(f"  Val samples:   {val_size}")

    # ---- Create model ----
    model = create_rehab_model(
        in_channels=3,
        graph_layout=args.graph_layout,
        graph_strategy=args.graph_strategy,
        lstm_hidden=args.lstm_hidden,
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # ---- Loss, optimizer, scheduler ----
    criterion = RehabLoss(delta=1.0)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # ---- Training loop ----
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\n{'─'*60}")
    print(f"  {'Epoch':>5} │ {'Train Loss':>10} │ {'Val Loss':>10} │ {'Val MAE':>8} │ {'LR':>8}")
    print(f"{'─'*60}")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion,
                                      optimizer, device)

        # Evaluate
        val_loss, val_mae, sample_attn = evaluate(model, val_loader,
                                                    criterion, device)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Print progress
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            improved = " ★"

            # Save best model
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
            }, save_path)
        else:
            patience_counter += 1

        print(f"  {epoch:5d} │ {train_loss:10.4f} │ {val_loss:10.4f} │ "
              f"{val_mae:8.4f} │ {current_lr:8.6f}{improved}")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n  ⚠ Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    # ---- Final results ----
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {os.path.join(args.save_dir, 'best_model.pth')}")
    print(f"{'='*60}")

    # ---- Show attention visualization ----
    if sample_attn is not None:
        visualize_attention(
            sample_attn,
            save_path=os.path.join(args.save_dir, 'sample_attention.npy')
        )

    # ---- Demo: predict on a single sample ----
    print(f"\n  Demo: Single Sample Prediction")
    print(f"  {'─'*40}")

    model.eval()
    with torch.no_grad():
        sample_data, sample_score, sample_len = dataset[0]
        sample_data = sample_data.unsqueeze(0).to(device)
        sample_len = sample_len.unsqueeze(0).to(device)

        pred, attn = model(sample_data, sample_len)
        print(f"  Ground Truth Score: {sample_score.item():.2f}")
        print(f"  Predicted Score:    {pred.item():.2f}")
        print(f"  Attention weights shape: {attn.shape}")
        print(f"  Top-5 attended timesteps: "
              f"{torch.topk(attn[0], min(5, attn.size(1))).indices.tolist()}")


if __name__ == '__main__':
    main()
