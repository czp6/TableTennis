"""
Multi-model pose detector with automatic fallback.
Tries multiple YOLOv8-Pose models in order of preference.
"""

import numpy as np
from typing import Optional, List
from pose_data import PoseKeypoints


class MultiModelPoseDetector:
    """Pose detector with multi-model fallback support."""

    def __init__(self):
        """Initialize multi-model detector with fallback chain."""
        self.models = []
        self.active_model = None
        self.model_names = []

        # Try loading models in order of preference (best to worst)
        model_configs = [
            ("yolov8m-pose.pt", "YOLOv8m-pose (medium)", 1),
            ("yolov8s-pose.pt", "YOLOv8s-pose (small)", 2),
            ("yolov8n-pose.pt", "YOLOv8n-pose (nano)", 3),
        ]

        for model_path, model_name, priority in model_configs:
            success = self._try_load_model(model_path, model_name, priority)
            if success and self.active_model is None:
                self.active_model = model_name

        if not self.models:
            print("Warning: No pose detection models could be loaded")
            self.available = False
        else:
            print(f"Loaded {len(self.models)} pose detection model(s)")
            print(f"Active model: {self.active_model}")
            self.available = True

    def _try_load_model(self, model_path: str, model_name: str, priority: int) -> bool:
        """
        Try to load a specific model.

        Args:
            model_path: Path to model file
            model_name: Human-readable model name
            priority: Priority level (lower = higher priority)

        Returns:
            True if model loaded successfully
        """
        try:
            from ultralytics import YOLO
            import torch

            model = YOLO(model_path)

            # Enable Mac GPU (MPS) acceleration if available
            if torch.backends.mps.is_available():
                model.to('mps')
                print(f"✓ Loaded {model_name} with Mac GPU acceleration (MPS)")
            else:
                print(f"✓ Loaded {model_name} (CPU mode)")

            self.models.append({
                'model': model,
                'name': model_name,
                'path': model_path,
                'priority': priority
            })
            self.model_names.append(model_name)
            return True
        except Exception as e:
            print(f"✗ Could not load {model_name}: {e}")
            return False

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        keypoint_conf_threshold: float = 0.3
    ) -> Optional[PoseKeypoints]:
        """
        Detect pose using available models with fallback.

        Args:
            frame: Input frame (BGR)
            conf_threshold: Confidence threshold for person detection
            keypoint_conf_threshold: Minimum confidence for individual keypoints

        Returns:
            PoseKeypoints object or None if no person detected
        """
        if not self.available or not self.models:
            return None

        # Try each model in order until one succeeds
        for model_info in self.models:
            try:
                result = self._detect_with_model(
                    model_info['model'],
                    frame,
                    conf_threshold,
                    keypoint_conf_threshold
                )
                if result is not None:
                    return result
            except Exception as e:
                # If this model fails, try the next one
                print(f"Warning: {model_info['name']} failed: {e}")
                continue

        return None

    def _detect_with_model(
        self,
        model,
        frame: np.ndarray,
        conf_threshold: float,
        keypoint_conf_threshold: float
    ) -> Optional[PoseKeypoints]:
        """
        Detect pose with a specific model.

        Args:
            model: YOLO model instance
            frame: Input frame
            conf_threshold: Detection confidence threshold
            keypoint_conf_threshold: Keypoint confidence threshold

        Returns:
            PoseKeypoints or None
        """
        # Run inference
        results = model(frame, conf=conf_threshold, verbose=False)

        # Get detections
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints_data = results[0].keypoints

            # Check if any person was detected
            if keypoints_data.data.shape[0] == 0:
                return None

            # Get the first person (highest confidence)
            person_keypoints = keypoints_data.data[0].cpu().numpy()

            # Filter keypoints by confidence
            for i in range(person_keypoints.shape[0]):
                if person_keypoints[i, 2] < keypoint_conf_threshold:
                    person_keypoints[i] = [0, 0, 0]

            # Check if we have enough valid keypoints
            valid_keypoints = np.sum(person_keypoints[:, 2] > keypoint_conf_threshold)
            if valid_keypoints < 5:
                return None

            # Create PoseKeypoints object
            pose_kp = PoseKeypoints(
                frame_idx=0,  # Will be updated by caller
                timestamp=0.0,  # Will be updated by caller
                keypoints=person_keypoints
            )

            return pose_kp

        return None

    def detect_multiple(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        keypoint_conf_threshold: float = 0.3,
        max_people: int = 2
    ) -> List[PoseKeypoints]:
        """
        Detect multiple people in frame.

        Args:
            frame: Input frame (BGR)
            conf_threshold: Confidence threshold for person detection
            keypoint_conf_threshold: Minimum confidence for individual keypoints
            max_people: Maximum number of people to detect

        Returns:
            List of PoseKeypoints objects
        """
        if not self.available or not self.models:
            return []

        # Use the first (best) available model
        model_info = self.models[0]

        try:
            # Run inference
            results = model_info['model'](frame, conf=conf_threshold, verbose=False)

            poses = []
            if len(results) > 0 and results[0].keypoints is not None:
                keypoints_data = results[0].keypoints

                # Get all detected people (up to max_people)
                num_people = min(keypoints_data.data.shape[0], max_people)

                for i in range(num_people):
                    person_keypoints = keypoints_data.data[i].cpu().numpy()

                    # Filter keypoints by confidence
                    for j in range(person_keypoints.shape[0]):
                        if person_keypoints[j, 2] < keypoint_conf_threshold:
                            person_keypoints[j] = [0, 0, 0]

                    # Check if we have enough valid keypoints
                    valid_keypoints = np.sum(person_keypoints[:, 2] > keypoint_conf_threshold)
                    if valid_keypoints < 5:
                        continue

                    pose_kp = PoseKeypoints(
                        frame_idx=0,  # Will be updated by caller
                        timestamp=0.0,  # Will be updated by caller
                        keypoints=person_keypoints
                    )
                    poses.append(pose_kp)

            return poses

        except Exception as e:
            print(f"Warning: Multi-person detection failed: {e}")
            return []

    def get_active_model_name(self) -> str:
        """Get the name of the currently active model."""
        return self.active_model if self.active_model else "None"

    def get_available_models(self) -> List[str]:
        """Get list of all available model names."""
        return self.model_names
