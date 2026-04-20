"""
==============================================================================
  Rehab Feeder: Dataset for Rehabilitation Exercise Assessment
==============================================================================

Handles:
  - Loading skeleton data with continuous quality scores (regression labels)
  - Variable-length sequences with zero-padding + length tracking
  - Data augmentation (optional random cropping and shifting)

Data format:
  - data_path: path to .npy file with shape (N, C, T, V, M)
  - score_path: path to .npy or .pkl file with continuous scores per sample

Usage:
  dataset = RehabFeeder(data_path='data.npy', score_path='scores.npy')
  data, score, length = dataset[0]
==============================================================================
"""

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset


class RehabFeeder(Dataset):
    """
    Dataset for skeleton-based rehabilitation exercise assessment.

    Unlike the original Feeder (classification labels), this uses
    continuous scores (float) for regression.

    Arguments:
        data_path:  path to .npy data, shape (N, C, T, V, M)
        score_path: path to scores file (.npy or .pkl)
                    - .npy: array of shape (N,) with float scores
                    - .pkl: list of float scores
        window_size: if > 0, pad/crop all sequences to this length
        random_choose: randomly choose a portion of sequence (augmentation)
        random_shift: randomly shift sequence in time (augmentation)
        debug: if True, only use first 100 samples
    """

    def __init__(self,
                 data_path,
                 score_path,
                 window_size=300,
                 random_choose=False,
                 random_shift=False,
                 debug=False,
                 mmap=True):

        self.data_path = data_path
        self.score_path = score_path
        self.window_size = window_size
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.debug = debug

        self.load_data(mmap)

    def load_data(self, mmap):
        """Load skeleton data and regression scores."""

        # ---- Load skeleton data ----
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # ---- Load scores (continuous values) ----
        if self.score_path.endswith('.npy'):
            self.scores = np.load(self.score_path).astype(np.float32)
        elif self.score_path.endswith('.pkl'):
            with open(self.score_path, 'rb') as f:
                self.scores = np.array(pickle.load(f), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported score file format: {self.score_path}")

        # ---- Debug mode: use only 100 samples ----
        if self.debug:
            self.data = self.data[:100]
            self.scores = self.scores[:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape
        print(f"[RehabFeeder] Loaded {self.N} samples | "
              f"Shape: ({self.C}, {self.T}, {self.V}, {self.M}) | "
              f"Score range: [{self.scores.min():.2f}, {self.scores.max():.2f}]")

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        """
        Returns:
            data: (C, T, V, M) skeleton tensor
            score: float — exercise quality score
            length: int — actual (non-padded) sequence length
        """
        data = np.array(self.data[index])    # (C, T, V, M)
        score = self.scores[index]

        # ---- Compute actual sequence length (non-zero frames) ----
        # A frame is "real" if it has any non-zero joint coordinates
        frame_energy = np.abs(data).sum(axis=(0, 2, 3))  # sum over C, V, M → (T,)
        actual_length = int((frame_energy > 0).sum())
        actual_length = max(actual_length, 1)  # at least 1 frame

        # ---- Data augmentation ----
        if self.random_choose and actual_length > self.window_size:
            # Randomly pick a sub-sequence
            start = np.random.randint(0, actual_length - self.window_size)
            data = data[:, start:start + self.window_size, :, :]
            actual_length = self.window_size

        if self.random_shift:
            # Randomly shift sequence position (small temporal jitter)
            shift = np.random.randint(-10, 10)
            data = np.roll(data, shift, axis=1)

        # ---- Pad or crop to window_size ----
        if self.window_size > 0:
            if data.shape[1] < self.window_size:
                # Zero-pad to window_size
                pad_length = self.window_size - data.shape[1]
                data = np.pad(data,
                              ((0, 0), (0, pad_length), (0, 0), (0, 0)),
                              mode='constant', constant_values=0)
            elif data.shape[1] > self.window_size:
                # Crop to window_size
                data = data[:, :self.window_size, :, :]
                actual_length = min(actual_length, self.window_size)

        return (torch.tensor(data, dtype=torch.float32),
                torch.tensor(score, dtype=torch.float32),
                torch.tensor(actual_length, dtype=torch.long))


def rehab_collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Handles batching of variable-length sequences.

    Returns:
        data: (N, C, T, V, M) — batched skeleton data
        scores: (N, 1) — batched scores
        lengths: (N,) — actual sequence lengths
    """
    data_list, score_list, length_list = zip(*batch)

    data = torch.stack(data_list, dim=0)           # (N, C, T, V, M)
    scores = torch.stack(score_list, dim=0).unsqueeze(1)  # (N, 1)
    lengths = torch.stack(length_list, dim=0)      # (N,)

    return data, scores, lengths


# =============================================================================
# Synthetic dataset generator (for testing without real data)
# =============================================================================
def create_synthetic_dataset(num_samples=100, num_frames=300,
                              num_joints=18, num_channels=3,
                              save_dir='data/rehab_synthetic'):
    """
    Creates a synthetic rehabilitation dataset for testing.

    The quality score is correlated with movement smoothness
    (lower variance = higher score) to simulate realistic data.

    Args:
        num_samples: number of exercise samples
        num_frames: frames per sample
        num_joints: number of skeleton joints
        num_channels: input channels (x, y, confidence)
        save_dir: directory to save generated data

    Returns:
        data_path: path to saved skeleton data
        score_path: path to saved scores
    """
    os.makedirs(save_dir, exist_ok=True)

    data = np.zeros((num_samples, num_channels, num_frames, num_joints, 1),
                    dtype=np.float32)
    scores = np.zeros(num_samples, dtype=np.float32)

    for i in range(num_samples):
        # Simulate skeleton movement with varying quality
        quality = np.random.uniform(0, 10)  # score: 0 (poor) to 10 (excellent)

        # Higher quality = smoother movement (less noise)
        noise_level = 0.5 - (quality / 20.0)  # quality 10 → noise 0.0
        noise_level = max(noise_level, 0.01)

        # Random actual length (variable-length sequences)
        actual_len = np.random.randint(100, num_frames)

        # Generate smooth base motion + noise
        t = np.linspace(0, 2 * np.pi, actual_len)
        for j in range(num_joints):
            for c in range(num_channels):
                base_motion = np.sin(t + j * 0.1 + c * 0.5)
                noise = np.random.randn(actual_len) * noise_level
                data[i, c, :actual_len, j, 0] = base_motion + noise

        scores[i] = quality

    # Save data
    data_path = os.path.join(save_dir, 'skeleton_data.npy')
    score_path = os.path.join(save_dir, 'scores.npy')
    np.save(data_path, data)
    np.save(score_path, scores)

    print(f"[Synthetic] Created {num_samples} samples → {data_path}")
    print(f"[Synthetic] Score range: [{scores.min():.2f}, {scores.max():.2f}]")

    return data_path, score_path


if __name__ == '__main__':
    # Generate and test synthetic data
    data_path, score_path = create_synthetic_dataset(num_samples=50)

    dataset = RehabFeeder(data_path, score_path, window_size=300)
    data, score, length = dataset[0]

    print(f"\nSample 0:")
    print(f"  Data shape: {data.shape}")
    print(f"  Score: {score.item():.2f}")
    print(f"  Actual length: {length.item()}")
