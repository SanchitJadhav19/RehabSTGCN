"""
==============================================================================
  Generate Test Data Script
==============================================================================

Generates a small synthetic dataset for testing the RehabSTGCN model.
Creates:
  - test_data/skeleton_test.npy
  - test_data/scores_test.npy

Usage:
  python generate_test_data.py
  python predict_rehab.py --data_path test_data/skeleton_test.npy --score_path test_data/scores_test.npy --index 0
==============================================================================
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feeder.feeder_rehab import create_synthetic_dataset

def main():
    test_dir = 'test_samples'
    print(f"\n--- Generating New Test Data in '{test_dir}/' ---")
    
    # Generate 5 fresh samples
    data_path, score_path = create_synthetic_dataset(
        num_samples=5, 
        num_frames=300, 
        num_joints=18, 
        save_dir=test_dir
    )
    
    # Rename to make it clearer for the user
    new_data_path = os.path.join(test_dir, 'new_exercise_samples.npy')
    new_score_path = os.path.join(test_dir, 'new_exercise_scores.npy')
    
    if os.path.exists(data_path):
        os.rename(data_path, new_data_path)
    if os.path.exists(score_path):
        os.rename(score_path, new_score_path)

    print(f"\n✓ Done! Created 5 new test samples.")
    print(f"  Data:   {new_data_path}")
    print(f"  Scores: {new_score_path}")
    print(f"\nTo check Sample #0, run this command:")
    print(f"  python predict_rehab.py --data_path {new_data_path} --score_path {new_score_path} --index 0")

if __name__ == '__main__':
    main()
