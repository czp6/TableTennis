"""
Temporal pose tracker for smooth, continuous pose detection.
Maintains pose state across frames and interpolates missing detections.
"""

import numpy as np
from typing import Optional, Deque
from collections import deque
from pose_data import PoseKeypoints


class TemporalPoseTracker:
    """Tracks poses across frames with temporal smoothing and interpolation."""

    def __init__(self, smoothing_window: int = 7, max_missing_frames: int = 15):
        """
        Initialize temporal pose tracker.

        Args:
            smoothing_window: Number of frames to use for smoothing (default: 7)
            max_missing_frames: Maximum frames to interpolate before giving up (default: 15)
        """
        self.smoothing_window = smoothing_window
        self.max_missing_frames = max_missing_frames

        # History of detected poses
        self.pose_history: Deque[Optional[PoseKeypoints]] = deque(maxlen=smoothing_window)

        # Track consecutive missing frames
        self.missing_frame_count = 0

        # Last valid pose for interpolation
        self.last_valid_pose: Optional[PoseKeypoints] = None

    def update(self, detected_pose: Optional[PoseKeypoints], frame_idx: int, timestamp: float) -> Optional[PoseKeypoints]:
        """
        Update tracker with new detection and return smoothed/interpolated pose.

        Args:
            detected_pose: Newly detected pose (or None if detection failed)
            frame_idx: Current frame index
            timestamp: Current timestamp

        Returns:
            Smoothed or interpolated pose, or None if tracking lost
        """
        if detected_pose is not None:
            # Valid detection - reset missing frame counter
            self.missing_frame_count = 0
            self.last_valid_pose = detected_pose
            self.pose_history.append(detected_pose)

            # Apply temporal smoothing
            smoothed_pose = self._smooth_pose(detected_pose, frame_idx, timestamp)
            return smoothed_pose

        else:
            # Missing detection - try to interpolate
            self.missing_frame_count += 1

            if self.missing_frame_count <= self.max_missing_frames and self.last_valid_pose is not None:
                # Interpolate based on recent history
                interpolated_pose = self._interpolate_pose(frame_idx, timestamp)
                self.pose_history.append(None)  # Mark as interpolated
                return interpolated_pose
            else:
                # Too many missing frames - tracking lost
                self.pose_history.append(None)
                return None

    def _smooth_pose(self, current_pose: PoseKeypoints, frame_idx: int, timestamp: float) -> PoseKeypoints:
        """
        Apply temporal smoothing to reduce jitter.

        Args:
            current_pose: Current detected pose
            frame_idx: Frame index
            timestamp: Timestamp

        Returns:
            Smoothed pose
        """
        # Get valid poses from history (excluding None/interpolated)
        valid_poses = [p for p in self.pose_history if p is not None]

        if len(valid_poses) < 2:
            # Not enough history - return current pose
            return current_pose

        # Smooth each keypoint using weighted average
        smoothed_keypoints = np.copy(current_pose.keypoints)

        for kp_idx in range(17):
            # Collect keypoint positions from history
            positions = []
            confidences = []
            weights = []

            for i, pose in enumerate(valid_poses):
                kp = pose.keypoints[kp_idx]
                if kp[2] > 0.15:  # Only use keypoints with some confidence (lowered threshold)
                    positions.append(kp[:2])
                    confidences.append(kp[2])
                    # More recent frames get higher weight
                    weight = (i + 1) / len(valid_poses)
                    weights.append(weight)

            # Add current detection with highest weight
            current_kp = current_pose.keypoints[kp_idx]
            if current_kp[2] > 0.15:
                positions.append(current_kp[:2])
                confidences.append(current_kp[2])
                weights.append(2.0)  # Current frame gets double weight

            if len(positions) > 0:
                # Weighted average of positions
                positions = np.array(positions)
                weights = np.array(weights)
                confidences = np.array(confidences)

                # Normalize weights
                weights = weights / weights.sum()

                # Compute weighted average
                smoothed_pos = np.average(positions, axis=0, weights=weights)
                smoothed_conf = np.average(confidences, weights=weights)

                smoothed_keypoints[kp_idx] = [smoothed_pos[0], smoothed_pos[1], smoothed_conf]

        # Create smoothed pose
        smoothed_pose = PoseKeypoints(
            frame_idx=frame_idx,
            timestamp=timestamp,
            keypoints=smoothed_keypoints
        )

        return smoothed_pose

    def _interpolate_pose(self, frame_idx: int, timestamp: float) -> Optional[PoseKeypoints]:
        """
        Interpolate pose when detection fails.

        Args:
            frame_idx: Frame index
            timestamp: Timestamp

        Returns:
            Interpolated pose or None
        """
        if self.last_valid_pose is None:
            return None

        # Get the two most recent valid poses for velocity estimation
        valid_poses = [p for p in self.pose_history if p is not None]

        if len(valid_poses) < 2:
            # Not enough history - just return last valid pose with reduced confidence
            interpolated_keypoints = np.copy(self.last_valid_pose.keypoints)
            # Reduce confidence based on how many frames we've missed (slower decay for smoother tracking)
            confidence_decay = 0.95 ** self.missing_frame_count
            interpolated_keypoints[:, 2] *= confidence_decay

            return PoseKeypoints(
                frame_idx=frame_idx,
                timestamp=timestamp,
                keypoints=interpolated_keypoints
            )

        # Estimate velocity from last two valid poses
        prev_pose = valid_poses[-2]
        curr_pose = valid_poses[-1]

        interpolated_keypoints = np.copy(curr_pose.keypoints)

        for kp_idx in range(17):
            prev_kp = prev_pose.keypoints[kp_idx]
            curr_kp = curr_pose.keypoints[kp_idx]

            if prev_kp[2] > 0.15 and curr_kp[2] > 0.15:
                # Estimate velocity
                velocity = curr_kp[:2] - prev_kp[:2]

                # Extrapolate position (with damping for more conservative interpolation)
                damping_factor = 0.8  # Reduce velocity over time
                extrapolated_pos = curr_kp[:2] + velocity * self.missing_frame_count * damping_factor

                # Reduce confidence based on missing frames (slower decay)
                confidence_decay = 0.92 ** self.missing_frame_count
                extrapolated_conf = curr_kp[2] * confidence_decay

                interpolated_keypoints[kp_idx] = [
                    extrapolated_pos[0],
                    extrapolated_pos[1],
                    extrapolated_conf
                ]

        return PoseKeypoints(
            frame_idx=frame_idx,
            timestamp=timestamp,
            keypoints=interpolated_keypoints
        )

    def reset(self):
        """Reset tracker state."""
        self.pose_history.clear()
        self.missing_frame_count = 0
        self.last_valid_pose = None

    def get_tracking_quality(self) -> float:
        """
        Get current tracking quality score (0-1).

        Returns:
            Quality score based on recent detection success
        """
        if len(self.pose_history) == 0:
            return 0.0

        valid_count = sum(1 for p in self.pose_history if p is not None)
        return valid_count / len(self.pose_history)
