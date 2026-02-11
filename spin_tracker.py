"""
Table Tennis Ball Spin Rate Tracker
Core processing module for detecting and calculating spin rate from video.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

# Enable OpenCV optimizations for better performance
cv2.setUseOptimized(True)
cv2.setNumThreads(0)  # Use all available CPU threads

# Try to import enhanced segmentation and YOLO detector
try:
    from enhanced_segmentation import EnhancedColorSegmentation
    ENHANCED_SEGMENTATION_AVAILABLE = True
except ImportError:
    ENHANCED_SEGMENTATION_AVAILABLE = False

try:
    from yolo_detector import YOLOBallDetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from hybrid_detector import HybridBallDetector
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False


@dataclass
class SpinData:
    """Container for spin rate measurements."""
    frame_number: int
    rotation_angle: float  # degrees
    rps: float  # revolutions per second (magnitude)
    ball_position: Optional[Tuple[int, int]] = None
    ball_radius: Optional[int] = None
    orange_centroid: Optional[Tuple[int, int]] = None
    white_centroid: Optional[Tuple[int, int]] = None
    topspin_rps: float = 0.0  # positive = topspin, negative = backspin
    sidespin_rps: float = 0.0  # positive = right spin, negative = left spin
    direction: str = "none"  # overall direction description


class SpinTracker:
    """Tracks table tennis ball spin rate from video."""

    def __init__(self, video_path: str, fps: float = 240.0, rotate_90_cw: bool = False):
        """
        Initialize the spin tracker.

        Args:
            video_path: Path to the video file
            fps: Frames per second of the video
            rotate_90_cw: Whether to rotate frames 90 degrees clockwise before processing
        """
        self.video_path = video_path
        self.fps = fps
        self.rotate_90_cw = rotate_90_cw

        # Use hardware-accelerated video decoding on Mac
        self.cap = cv2.VideoCapture(video_path, cv2.CAP_AVFOUNDATION)

        # Fallback to default backend if AVFoundation fails
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Color ranges for orange and white in HSV
        self.orange_lower = np.array([5, 100, 100])
        self.orange_upper = np.array([25, 255, 255])
        # More permissive white detection (low saturation, high value)
        self.white_lower = np.array([0, 0, 180])
        self.white_upper = np.array([180, 50, 255])

        # Tracking state
        self.previous_angle = None
        self.total_rotation = 0.0
        self.total_horizontal_rotation = 0.0
        self.total_vertical_rotation = 0.0
        self.spin_history: List[SpinData] = []

        # Smoothing for RPS values (not angles)
        self.topspin_rps_history = []
        self.sidespin_rps_history = []
        self.smoothing_window = 10  # Smooth the output RPS values

        # Detection parameters (adjustable in real-time)
        self.yolo_weight = 1.0  # Weight for YOLOv8 detection (0-1)
        self.hough_weight = 0.0  # Weight for Hough Circle detection (0-1)
        self.hybrid_weight = 0.0  # Weight for Hybrid detection (0-1)
        self.yolo_conf_threshold = 0.25
        self.yolo_min_mask_area = 100  # Minimum mask area in pixels
        self.yolo_radius_scale = 0.8  # Circle radius scale factor (0.0-1.0)
        self.yolo_debug = False  # Debug mode for YOLO detector
        self.hough_param1 = 50
        self.hough_param2 = 30
        self.hough_min_radius = 10
        self.hough_max_radius = 150

        # Hybrid detector parameters
        self.hybrid_confidence_threshold = 0.6
        self.hybrid_orange_h_min = 5
        self.hybrid_orange_h_max = 25
        self.hybrid_orange_s_min = 100
        self.hybrid_orange_v_min = 100
        self.hybrid_white_v_min = 180
        self.hybrid_white_s_max = 50

        # Initialize YOLO detector if available
        self.yolo_detector = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_detector = YOLOBallDetector()
                print("YOLOv8 detector loaded")
            except Exception as e:
                print(f"Could not load YOLO detector: {e}")

        # Initialize hybrid detector if available
        self.hybrid_detector = None
        if HYBRID_AVAILABLE:
            try:
                self.hybrid_detector = HybridBallDetector()
                print("Hybrid detector loaded")
            except Exception as e:
                print(f"Could not load Hybrid detector: {e}")

        # Initialize enhanced color segmentation
        self.color_segmenter = None
        if ENHANCED_SEGMENTATION_AVAILABLE:
            self.color_segmenter = EnhancedColorSegmentation()
            print("Enhanced color segmentation enabled")

    def detect_ball(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Detect the ball in the frame using YOLOv8, Hough Circles, and/or Hybrid detection with weighted combination.

        Args:
            frame: Input frame

        Returns:
            Tuple of (x, y, radius) or None if not found
        """
        yolo_result = None
        hough_result = None
        hybrid_result = None

        # Try YOLO detection
        if self.yolo_detector and self.yolo_detector.available and self.yolo_weight > 0:
            yolo_result = self.yolo_detector.detect(
                frame,
                conf_threshold=self.yolo_conf_threshold,
                min_mask_area=self.yolo_min_mask_area,
                radius_scale=self.yolo_radius_scale,
                debug=self.yolo_debug
            )

        # Try Hybrid detection (Propose & Validate)
        if self.hybrid_detector and self.hybrid_weight > 0:
            # Update hybrid detector parameters
            self.hybrid_detector.confidence_threshold = self.hybrid_confidence_threshold
            self.hybrid_detector.hough_param1 = self.hough_param1
            self.hybrid_detector.hough_param2 = self.hough_param2
            self.hybrid_detector.hough_min_radius = self.hough_min_radius
            self.hybrid_detector.hough_max_radius = self.hough_max_radius

            # Update color validator parameters
            self.hybrid_detector.color_validator.orange_lower = np.array([
                self.hybrid_orange_h_min,
                self.hybrid_orange_s_min,
                self.hybrid_orange_v_min
            ])
            self.hybrid_detector.color_validator.orange_upper = np.array([
                self.hybrid_orange_h_max,
                255,
                255
            ])
            self.hybrid_detector.color_validator.white_lower = np.array([
                0,
                0,
                self.hybrid_white_v_min
            ])
            self.hybrid_detector.color_validator.white_upper = np.array([
                180,
                self.hybrid_white_s_max,
                255
            ])

            hybrid_result = self.hybrid_detector.detect(frame)

        # Try Hough Circle detection
        if self.hough_weight > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)

            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=self.hough_param1,
                param2=self.hough_param2,
                minRadius=self.hough_min_radius,
                maxRadius=self.hough_max_radius
            )

            if circles is not None:
                circles = np.uint16(np.around(circles))
                x, y, r = circles[0][0]
                hough_result = (int(x), int(y), int(r))

        # Combine results based on weights
        results = []
        weights = []

        if yolo_result and self.yolo_weight > 0:
            results.append(yolo_result)
            weights.append(self.yolo_weight)

        if hybrid_result and self.hybrid_weight > 0:
            results.append(hybrid_result)
            weights.append(self.hybrid_weight)

        if hough_result and self.hough_weight > 0:
            results.append(hough_result)
            weights.append(self.hough_weight)

        if not results:
            return None

        if len(results) == 1:
            return results[0]

        # Weighted average of all detections
        total_weight = sum(weights)
        combined_x = sum(r[0] * w for r, w in zip(results, weights)) / total_weight
        combined_y = sum(r[1] * w for r, w in zip(results, weights)) / total_weight
        combined_r = sum(r[2] * w for r, w in zip(results, weights)) / total_weight

        return (int(combined_x), int(combined_y), int(combined_r))

    def _detect_ball_ml_only(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Detect ball using ML model only (no OpenCV Hough Circles).
        Uses sliding window approach with the ML model.

        Args:
            frame: Input frame

        Returns:
            Tuple of (x, y, radius) or None if not found
        """
        if not self.ml_detector:
            return None

        # Use previous position if available to limit search area
        if self.previous_ball_pos:
            px, py, pr = self.previous_ball_pos
            # Search in expanded region
            margin = int(pr * 5)
            x_start = max(0, px - margin)
            y_start = max(0, py - margin)
            x_end = min(frame.shape[1], px + margin)
            y_end = min(frame.shape[0], py + margin)
            search_step = max(10, pr // 2)
        else:
            # Full frame search with coarse grid
            x_start, y_start = 0, 0
            x_end, y_end = frame.shape[1], frame.shape[0]
            search_step = 30

        # Estimate radius from previous detection or use default
        estimated_radius = self.previous_ball_pos[2] if self.previous_ball_pos else 40

        # Sliding window search
        best_position = None
        best_confidence = 0.0

        for y in range(y_start, y_end, search_step):
            for x in range(x_start, x_end, search_step):
                # Test this position
                confidence = self.ml_detector.validate_with_ml(frame, (x, y, estimated_radius))

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_position = (x, y, estimated_radius)

        # Accept if confidence is above threshold
        if best_position and best_confidence > 0.5:
            self.previous_ball_pos = best_position
            self.ball_position_history.append(best_position)
            if len(self.ball_position_history) > self.max_position_history:
                self.ball_position_history.pop(0)
            return best_position

        return None

    def _validate_ball_colors(self, frame: np.ndarray, ball_pos: Tuple[int, int, int]) -> float:
        """
        Validate that a detected circle contains orange and/or white colors.

        Args:
            frame: Input frame
            ball_pos: Tuple of (x, y, radius)

        Returns:
            Score between 0 and 1 indicating confidence this is the ball
        """
        x, y, r = ball_pos

        # Extract ball region
        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(frame.shape[1], x + r)
        y2 = min(frame.shape[0], y + r)

        ball_region = frame[y1:y2, x1:x2]

        if ball_region.size == 0:
            return 0.0

        # Convert to HSV
        hsv = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)

        # Check for orange - more permissive range
        orange_mask = cv2.inRange(hsv, self.orange_lower, self.orange_upper)
        orange_ratio = np.count_nonzero(orange_mask) / orange_mask.size

        # Check for white - more permissive
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        white_ratio = np.count_nonzero(white_mask) / white_mask.size

        # Check for skin tones (to reject hands) - HSV ranges for skin
        skin_lower = np.array([0, 20, 70])
        skin_upper = np.array([20, 150, 255])
        skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
        skin_ratio = np.count_nonzero(skin_mask) / skin_mask.size

        # If it's mostly skin-colored, reject it
        if skin_ratio > 0.4 and orange_ratio < 0.15:
            return 0.0

        # Score based on presence of orange or white (ball should have at least one)
        # Give higher weight to orange since it's more distinctive
        score = (orange_ratio * 2.0) + white_ratio

        return min(score, 1.0)

    def calculate_rotation_angle_from_mask(self, frame: np.ndarray, ball_pos: Tuple[int, int, int], segmentation_mask: Optional[np.ndarray] = None) -> Optional[Tuple[float, Tuple[int, int], Tuple[int, int]]]:
        """
        Calculate rotation angle using green circle center and internal color analysis.

        Args:
            frame: Input frame
            ball_pos: Tuple of (x, y, radius) from detection (green circle)
            segmentation_mask: YOLO segmentation mask (not used, kept for compatibility)

        Returns:
            Tuple of (angle_degrees, ball_center, orange_centroid) or None
        """
        # Step 1: Use green circle center as ball center
        x, y, r = ball_pos
        ball_center = (x, y)

        # Step 2: Create circular mask using the green circle
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        # Extract ROI
        x_min = max(0, x - r)
        y_min = max(0, y - r)
        x_max = min(frame.shape[1], x + r)
        y_max = min(frame.shape[0], y + r)

        roi = frame[y_min:y_max, x_min:x_max]
        roi_mask = mask[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            return None

        # Step 3: Apply HSV color mask to find orange pixels within the circular region
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Create orange mask with wrap-around for red
        # Lower red range (0-10)
        lower_red_lower = np.array([0, 100, 100])
        lower_red_upper = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_roi, lower_red_lower, lower_red_upper)

        # Orange range (5-25)
        orange_mask = cv2.inRange(hsv_roi, self.orange_lower, self.orange_upper)

        # Upper red range (170-180)
        upper_red_lower = np.array([170, 100, 100])
        upper_red_upper = np.array([180, 255, 255])
        mask3 = cv2.inRange(hsv_roi, upper_red_lower, upper_red_upper)

        # Combine all orange/red ranges
        combined_orange_mask = cv2.bitwise_or(cv2.bitwise_or(mask1, orange_mask), mask3)

        # Apply the circular mask to only consider pixels inside the green circle
        combined_orange_mask = cv2.bitwise_and(combined_orange_mask, roi_mask)

        # Step 4: Find centroid of orange region
        orange_moments = cv2.moments(combined_orange_mask)

        if orange_moments['m00'] == 0:
            return None

        orange_cx_roi = int(orange_moments['m10'] / orange_moments['m00'])
        orange_cy_roi = int(orange_moments['m01'] / orange_moments['m00'])

        # Convert back to full frame coordinates
        orange_cx = orange_cx_roi + x_min
        orange_cy = orange_cy_roi + y_min
        orange_centroid = (orange_cx, orange_cy)

        # Step 5: Calculate vector from ball center (green circle) to orange centroid
        dx = orange_cx - x
        dy = orange_cy - y

        # Step 6: Calculate angle using atan2
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        # Normalize to 0-360 range
        if angle_deg < 0:
            angle_deg += 360

        return (angle_deg, ball_center, orange_centroid)
        """
        Calculate the rotation angle of the ball based on color distribution.
        Uses enhanced color segmentation if available.

        Args:
            frame: Input frame
            ball_pos: Tuple of (x, y, radius)

        Returns:
            Tuple of (angle_degrees, orange_centroid, white_centroid, vis_frame) or None if calculation fails
        """
        # Use enhanced color segmentation if available
        if self.color_segmenter:
            orange_cent, white_cent, vis_frame = self.color_segmenter.segment_orange_white(frame, ball_pos)

            if orange_cent is None:
                return None

            # Calculate angle
            x, y, r = ball_pos
            angle = self.color_segmenter.calculate_rotation_angle((x, y), orange_cent, white_cent)

            if angle is None:
                return None

            # Return with boundary curve
            return (angle, orange_cent, white_cent, vis_frame)

        # Fallback to original method
        x, y, r = ball_pos

        # Create a mask for the ball region
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        # Extract ball region
        ball_region = cv2.bitwise_and(frame, frame, mask=mask)
        hsv = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)

        # Create mask for orange
        orange_mask = cv2.inRange(hsv, self.orange_lower, self.orange_upper)
        orange_mask = cv2.bitwise_and(orange_mask, mask)

        # Create white mask by subtracting orange from the ball
        # This is more reliable than color detection for white
        white_mask = cv2.bitwise_and(mask, cv2.bitwise_not(orange_mask))

        # Find contours for both colors
        orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get largest contours
        orange_contour = max(orange_contours, key=cv2.contourArea) if orange_contours else None
        white_contour = max(white_contours, key=cv2.contourArea) if white_contours else None

        # Find the centroid of orange region
        orange_moments = cv2.moments(orange_mask)
        white_moments = cv2.moments(white_mask)

        orange_centroid = None
        white_centroid = None

        if orange_moments['m00'] > 0:
            orange_cx = int(orange_moments['m10'] / orange_moments['m00'])
            orange_cy = int(orange_moments['m01'] / orange_moments['m00'])
            orange_centroid = (orange_cx, orange_cy)

            # Calculate overall angle from ball center to orange centroid
            angle = np.arctan2(orange_cy - y, orange_cx - x)
            angle_degrees = np.degrees(angle)

            if white_moments['m00'] > 0:
                white_cx = int(white_moments['m10'] / white_moments['m00'])
                white_cy = int(white_moments['m01'] / white_moments['m00'])
                white_centroid = (white_cx, white_cy)

            return (angle_degrees, orange_centroid, white_centroid, orange_mask, white_mask)

        return None

    def calculate_spin_rate(self, current_angle: float, frame_number: int) -> Tuple[float, float, float, str]:
        """
        Calculate spin rate based on angle change.

        Camera is assumed to be facing horizontally (perpendicular to ball trajectory):
        - Horizontal movement in frame = topspin/backspin
        - Vertical movement in frame = sidespin (left/right)

        Args:
            current_angle: Current overall rotation angle in degrees
            frame_number: Current frame number

        Returns:
            Tuple of (total_rps, topspin_rps, sidespin_rps, direction_description)
        """
        if self.previous_angle is None:
            self.previous_angle = current_angle
            return (0.0, 0.0, 0.0, "none")

        # Calculate angle difference
        angle_diff = current_angle - self.previous_angle

        # Handle angle wrapping (-180 to 180)
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360

        # Accumulate total rotation
        self.total_rotation += angle_diff

        # Calculate total RPS
        time_elapsed = frame_number / self.fps
        if time_elapsed > 0:
            total_rps = (self.total_rotation / 360.0) / time_elapsed
        else:
            total_rps = 0.0

        # Decompose into topspin/backspin and sidespin components
        angle_change_per_frame = angle_diff
        current_angle_rad = np.radians(current_angle)

        # Horizontal component (topspin/backspin)
        topspin_contribution = angle_change_per_frame * np.cos(current_angle_rad)

        # Vertical component (sidespin)
        sidespin_contribution = angle_change_per_frame * np.sin(current_angle_rad)

        self.total_horizontal_rotation += topspin_contribution
        self.total_vertical_rotation += sidespin_contribution

        if time_elapsed > 0:
            topspin_rps_raw = (self.total_horizontal_rotation / 360.0) / time_elapsed
            sidespin_rps_raw = (self.total_vertical_rotation / 360.0) / time_elapsed
        else:
            topspin_rps_raw = 0.0
            sidespin_rps_raw = 0.0

        # Smooth the RPS values
        self.topspin_rps_history.append(topspin_rps_raw)
        self.sidespin_rps_history.append(sidespin_rps_raw)

        if len(self.topspin_rps_history) > self.smoothing_window:
            self.topspin_rps_history.pop(0)
        if len(self.sidespin_rps_history) > self.smoothing_window:
            self.sidespin_rps_history.pop(0)

        # Use smoothed values
        topspin_rps = sum(self.topspin_rps_history) / len(self.topspin_rps_history)
        sidespin_rps = sum(self.sidespin_rps_history) / len(self.sidespin_rps_history)

        self.previous_angle = current_angle

        # Determine direction description with threshold
        direction_parts = []
        if abs(topspin_rps) > 0.5:
            if topspin_rps > 0:
                direction_parts.append("topspin")
            else:
                direction_parts.append("backspin")

        if abs(sidespin_rps) > 0.5:
            if sidespin_rps > 0:
                direction_parts.append("right-spin")
            else:
                direction_parts.append("left-spin")

        direction = " + ".join(direction_parts) if direction_parts else "minimal"

        return (abs(total_rps), topspin_rps, sidespin_rps, direction)

    def process_frame(self, frame: np.ndarray, frame_number: int) -> Optional[SpinData]:
        """
        Process a single frame to extract spin data.

        Args:
            frame: Input frame
            frame_number: Frame number

        Returns:
            SpinData object or None if processing fails
        """
        # Rotate frame if needed
        if self.rotate_90_cw:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        ball_pos = self.detect_ball(frame)

        if ball_pos is None:
            return None

        # Get segmentation mask from YOLO detector
        segmentation_mask = None
        if self.yolo_detector and self.yolo_detector.available:
            segmentation_mask = self.yolo_detector.get_last_mask()

        # Use new mask-based spin analysis
        result = self.calculate_rotation_angle_from_mask(frame, ball_pos, segmentation_mask)

        if result is None:
            return None

        angle, ball_center, orange_centroid = result

        total_rps, topspin_rps, sidespin_rps, direction = self.calculate_spin_rate(angle, frame_number)

        spin_data = SpinData(
            frame_number=frame_number,
            rotation_angle=angle,
            rps=total_rps,
            ball_position=ball_center,
            ball_radius=ball_pos[2],
            orange_centroid=orange_centroid,
            white_centroid=None,  # Not used in new method
            topspin_rps=topspin_rps,
            sidespin_rps=sidespin_rps,
            direction=direction
        )

        self.spin_history.append(spin_data)

        return spin_data

    def process_video(self, progress_callback=None) -> List[SpinData]:
        """
        Process the entire video.

        Args:
            progress_callback: Optional callback function(frame_num, total_frames)

        Returns:
            List of SpinData objects
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_number = 0

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            self.process_frame(frame, frame_number)

            if progress_callback:
                progress_callback(frame_number, self.frame_count)

            frame_number += 1

        return self.spin_history

    def get_average_spin(self) -> float:
        """Get average spin rate across all frames."""
        if not self.spin_history:
            return 0.0

        valid_spins = [s.rps for s in self.spin_history if abs(s.rps) > 0.1]

        if not valid_spins:
            return 0.0

        return sum(valid_spins) / len(valid_spins)

    def release(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
