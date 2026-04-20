"""
==============================================================================
  RehabPredictor — End-to-end Video → Score Pipeline
==============================================================================

Combines skeleton extraction, data preprocessing, and STGCN-LSTM inference
into a single class. This is the main entry point for the backend API.
"""

import os
import numpy as np
import joblib

from ml_model.graph import Graph
from ml_model.model import build_model
from ml_model.skeleton_extractor import extract_skeleton_from_video

# Joint indices in the 100-column raw skeleton row (25 joints × 4 values)
# Each joint has: X, Y, Z, Status (4 values)
index_Spine_Base = 0
index_Spine_Mid = 4
index_Neck = 8
index_Head = 12
index_Shoulder_Left = 16
index_Elbow_Left = 20
index_Wrist_Left = 24
index_Hand_Left = 28
index_Shoulder_Right = 32
index_Elbow_Right = 36
index_Wrist_Right = 40
index_Hand_Right = 44
index_Hip_Left = 48
index_Knee_Left = 52
index_Ankle_Left = 56
index_Foot_Left = 60
index_Hip_Right = 64
index_Knee_Right = 68
index_Ankle_Right = 72
index_Foot_Right = 76
index_Spine_Shoulder = 80
index_Tip_Left = 84
index_Thumb_Left = 88
index_Tip_Right = 92
index_Thumb_Right = 96

BODY_PART = [
    index_Spine_Base, index_Spine_Mid, index_Neck, index_Head,
    index_Shoulder_Left, index_Elbow_Left, index_Wrist_Left, index_Hand_Left,
    index_Shoulder_Right, index_Elbow_Right, index_Wrist_Right, index_Hand_Right,
    index_Hip_Left, index_Knee_Left, index_Ankle_Left, index_Foot_Left,
    index_Hip_Right, index_Knee_Right, index_Ankle_Right, index_Ankle_Right,
    index_Spine_Shoulder, index_Tip_Left, index_Thumb_Left,
    index_Tip_Right, index_Thumb_Right
]

NUM_JOINTS = 25
NUM_CHANNEL = 3


class RehabPredictor:
    """
    End-to-end predictor: Video → Skeleton → Score.
    
    Usage:
        predictor = RehabPredictor('/path/to/pretrained/')
        result = predictor.predict_from_video('/path/to/video.mp4')
    """

    def __init__(self, pretrained_dir=None):
        """
        Args:
            pretrained_dir: Directory containing best_model.hdf5, sc_x.save, sc_y.save
        """
        if pretrained_dir is None:
            pretrained_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'pretrained'
            )

        weights_path = os.path.join(pretrained_dir, 'best_model.hdf5')
        sc_x_path = os.path.join(pretrained_dir, 'sc_x.save')
        sc_y_path = os.path.join(pretrained_dir, 'sc_y.save')

        # Validate files exist
        for path, name in [(weights_path, 'Model weights'), 
                           (sc_x_path, 'Input scaler'),
                           (sc_y_path, 'Output scaler')]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found at: {path}")

        # Load scalers
        print("[RehabPredictor] Loading scalers...")
        self.sc_x = joblib.load(sc_x_path)
        self.sc_y = joblib.load(sc_y_path)

        # Build graph and model
        print("[RehabPredictor] Building STGCN-LSTM model...")
        graph = Graph(NUM_JOINTS)
        self.model = build_model(
            NUM_JOINTS, NUM_CHANNEL,
            graph.bias_mat_1, graph.bias_mat_2
        )

        # Load weights
        print(f"[RehabPredictor] Loading weights from {weights_path}...")
        self.model.load_weights(weights_path)
        print("[RehabPredictor] Model ready!")

    def _preprocess_skeleton(self, skeleton_data):
        """
        Preprocess raw skeleton data (from extractor) into model input format.
        
        Args:
            skeleton_data: np.array of shape (num_frames, 100) — 25 joints × 4 values
            
        Returns:
            X: np.array of shape (1, num_timestep, 25, 3) ready for model
        """
        num_timestep = skeleton_data.shape[0]
        batch_size = 1

        # Extract X, Y, Z for each body part (skip status column)
        X = np.zeros((num_timestep, NUM_JOINTS * NUM_CHANNEL), dtype='float32')
        for row in range(num_timestep):
            counter = 0
            for parts in BODY_PART:
                for i in range(NUM_CHANNEL):
                    X[row, counter + i] = skeleton_data[row, parts + i]
                counter += NUM_CHANNEL

        # Apply StandardScaler
        X = self.sc_x.transform(X)

        # Reshape to (batch, timestep, joints, channels)
        X_ = np.zeros((batch_size, num_timestep, NUM_JOINTS, NUM_CHANNEL))
        for timestep in range(num_timestep):
            for node in range(NUM_JOINTS):
                for channel in range(NUM_CHANNEL):
                    X_[0, timestep, node, channel] = X[
                        timestep, channel + (node * NUM_CHANNEL)
                    ]

        return X_

    def predict_from_skeleton(self, skeleton_data):
        """
        Run prediction on pre-extracted skeleton data.
        
        Args:
            skeleton_data: np.array of shape (num_frames, 100)
            
        Returns:
            score: float — predicted rehabilitation quality score
        """
        X = self._preprocess_skeleton(skeleton_data)
        y_pred = self.model.predict(X, verbose=0)
        score = self.sc_y.inverse_transform(y_pred)
        return float(score[0, 0])

    def predict_from_video(self, video_path):
        """
        Full pipeline: Video → Skeleton → Score.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            dict with:
                - score: float (predicted quality score)
                - num_frames: int (detected frames)
                - frame_confidences: list of floats
                - video_info: dict (fps, resolution, etc.)
        """
        # Step 1: Extract skeleton
        print(f"[RehabPredictor] Extracting skeleton from: {video_path}")
        skeleton_data, metadata = extract_skeleton_from_video(video_path)

        # Step 2: Run model
        print(f"[RehabPredictor] Running STGCN-LSTM inference...")
        score = self.predict_from_skeleton(skeleton_data)

        print(f"[RehabPredictor] Predicted score: {score:.2f}")

        return {
            'score': round(score, 2),
            'num_frames': metadata['detected_frames'],
            'frame_confidences': metadata.get('frame_confidences', []),
            'video_info': {
                'fps': metadata['fps'],
                'total_frames': metadata['total_frames'],
                'width': metadata['width'],
                'height': metadata['height'],
            }
        }
