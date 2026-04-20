"""
==============================================================================
  Video → Skeleton Extraction using MediaPipe (25-joint Kinect v2 mapping)
==============================================================================

Ported from STGCN-rehab-main/extract_skeleton.py.
Extracts 25 Kinect v2 joints from video using MediaPipe and returns
numpy arrays ready for the STGCN-LSTM model (instead of writing CSV).
"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose


def get_kinect_joints(landmarks):
    """
    Convert MediaPipe 33-point landmarks to 25-joint Kinect v2 format.
    Returns: np.array of shape (25, 3) with X, Y, Z coordinates.
    """
    points = {}
    for i, lm in enumerate(landmarks.landmark):
        points[i] = np.array([lm.x, lm.y, lm.z])
    
    def mid(p1, p2):
        return (points[p1] + points[p2]) / 2.0
    def mid_arr(arr1, arr2):
        return (arr1 + arr2) / 2.0

    # Compute derived joints
    spine_base = mid(23, 24)          # Midpoint of hips
    spine_shoulder = mid(11, 12)      # Midpoint of shoulders
    spine_mid = mid_arr(spine_base, spine_shoulder)
    neck = mid_arr(spine_shoulder, points[0])

    # Build 25-joint array in Kinect v2 order
    kj = np.zeros((25, 3))
    kj[0] = spine_base
    kj[1] = spine_mid
    kj[2] = neck
    kj[3] = points[0]   # Head/Nose
    kj[4] = points[11]  # ShoulderLeft
    kj[5] = points[13]  # ElbowLeft
    kj[6] = points[15]  # WristLeft
    kj[7] = mid(17, 19) # HandLeft
    kj[8] = points[12]  # ShoulderRight
    kj[9] = points[14]  # ElbowRight
    kj[10] = points[16] # WristRight
    kj[11] = mid(18, 20) # HandRight
    kj[12] = points[23]  # HipLeft
    kj[13] = points[25]  # KneeLeft
    kj[14] = points[27]  # AnkleLeft
    kj[15] = points[31]  # FootLeft
    kj[16] = points[24]  # HipRight
    kj[17] = points[26]  # KneeRight
    kj[18] = points[28]  # AnkleRight
    kj[19] = spine_shoulder  # SpineShoulder
    kj[20] = points[19]  # HandTipLeft
    kj[21] = points[21]  # ThumbLeft
    kj[22] = points[20]  # HandTipRight
    kj[23] = points[22]  # ThumbRight
    kj[24] = points[32]  # FootRight
    
    return kj


def extract_skeleton_from_video(video_path):
    """
    Extract 25-joint skeleton data from a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        skeleton_data: np.array of shape (num_frames, 100) 
                       [25 joints × 4 values (X, Y, Z, status)]
        metadata: dict with fps, total_frames, width, height, detected_frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    metadata = {
        'fps': fps,
        'total_frames': total_frames,
        'width': width,
        'height': height,
    }

    all_rows = []
    frame_confidences = []

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2
    ) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_world_landmarks:
                joints_25 = get_kinect_joints(results.pose_world_landmarks)
                
                row = []
                for joint in joints_25:
                    row.append(joint[0])  # X
                    row.append(joint[1])  # Y
                    row.append(joint[2])  # Z
                    row.append(2)         # Status (Tracked)
                
                all_rows.append(row)
                
                # Compute average visibility as frame confidence
                avg_visibility = np.mean([
                    lm.visibility for lm in results.pose_world_landmarks.landmark
                ])
                frame_confidences.append(float(avg_visibility))
            else:
                # No pose detected in this frame — skip it
                frame_confidences.append(0.0)

    cap.release()

    if len(all_rows) == 0:
        raise ValueError("No pose detected in any frame of the video")

    skeleton_data = np.array(all_rows, dtype=np.float32)
    metadata['detected_frames'] = len(all_rows)
    metadata['frame_confidences'] = frame_confidences

    print(f"[SkeletonExtractor] Extracted {len(all_rows)} frames from video")
    return skeleton_data, metadata
