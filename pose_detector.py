"""
YOLOv8-Pose detector wrapper for player pose detection.
"""

import cv2
import numpy as np
from typing import Optional
from pose_data import PoseKeypoints


class PoseDetector:
    """Wrapper for YOLOv8-Pose model to detect player poses."""

    def __init__(self, model_path: str = "yolov8m-pose.pt"):
        """
        Initialize pose detector.

        Args:
            model_path: Path to YOLOv8-Pose model (default: yolov8m-pose.pt)
        """
        try:
            from ultralytics import YOLO
            import torch

            self.model = YOLO(model_path)

            # Enable Mac GPU (MPS) acceleration if available
            if torch.backends.mps.is_available():
                self.model.to('mps')
                print(f"Loaded YOLOv8-Pose model from {model_path} with Mac GPU acceleration (MPS)")
            else:
                print(f"Loaded YOLOv8-Pose model from {model_path} (CPU mode)")

            self.available = True
        except ImportError:
            print("Warning: ultralytics not installed. Install with: pip install ultralytics")
            self.available = False
            self.model = None
        except Exception as e:
            print(f"Warning: Could not load YOLOv8-Pose model: {e}")
            self.available = False
            self.model = None

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.25, keypoint_conf_threshold: float = 0.3) -> Optional[PoseKeypoints]:
        """
        Detect pose in frame.

        Args:
            frame: Input frame (BGR)
            conf_threshold: Confidence threshold for person detection
            keypoint_conf_threshold: Minimum confidence for individual keypoints

        Returns:
            PoseKeypoints object or None if no person detected
        """
        if not self.available or self.model is None:
            return None

        # Run inference
        results = self.model(frame, conf=conf_threshold, verbose=False)

        # Get detections
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints_data = results[0].keypoints

            # Check if any person was detected
            if keypoints_data.data.shape[0] == 0:
                return None

            # Get the first person (highest confidence)
            # keypoints shape: [num_people, 17, 3] where 3 = (x, y, confidence)
            person_keypoints = keypoints_data.data[0].cpu().numpy()

            # Filter keypoints by confidence - set low confidence keypoints to zero
            for i in range(person_keypoints.shape[0]):
                if person_keypoints[i, 2] < keypoint_conf_threshold:
                    person_keypoints[i] = [0, 0, 0]

            # Check if we have enough valid keypoints (at least 5 for meaningful pose)
            valid_keypoints = np.sum(person_keypoints[:, 2] > keypoint_conf_threshold)
            if valid_keypoints < 5:
                return None

            # Create PoseKeypoints object
            # Note: frame_idx and timestamp will be set by the caller
            pose_kp = PoseKeypoints(
                frame_idx=0,  # Will be updated by caller
                timestamp=0.0,  # Will be updated by caller
                keypoints=person_keypoints
            )

            return pose_kp

        return None

    def detect_multiple(self, frame: np.ndarray, conf_threshold: float = 0.5, max_people: int = 2):
        """
        Detect multiple people in frame.

        Args:
            frame: Input frame (BGR)
            conf_threshold: Confidence threshold for keypoint detection
            max_people: Maximum number of people to detect

        Returns:
            List of PoseKeypoints objects
        """
        if not self.available or self.model is None:
            return []

        # Run inference
        results = self.model(frame, conf=conf_threshold, verbose=False)

        poses = []
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints_data = results[0].keypoints

            # Get all detected people (up to max_people)
            num_people = min(keypoints_data.data.shape[0], max_people)

            for i in range(num_people):
                person_keypoints = keypoints_data.data[i].cpu().numpy()

                pose_kp = PoseKeypoints(
                    frame_idx=0,  # Will be updated by caller
                    timestamp=0.0,  # Will be updated by caller
                    keypoints=person_keypoints
                )
                poses.append(pose_kp)

        return poses
