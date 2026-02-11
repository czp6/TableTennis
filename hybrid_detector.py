#!/usr/bin/env python3
"""
Hybrid ball detection using "Propose & Validate" approach.
Combines Hough Circle Transform (geometry proposer) with HSV color validation (pixel voting).
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


class ColorValidator:
    """Creates HSV color masks for orange and white with red wrap-around handling."""

    def __init__(self):
        # Orange HSV ranges (default values)
        self.orange_lower = np.array([5, 100, 100])
        self.orange_upper = np.array([25, 255, 255])

        # White HSV ranges
        self.white_lower = np.array([0, 0, 180])
        self.white_upper = np.array([180, 50, 255])

    def create_orange_mask(self, hsv_frame: np.ndarray) -> np.ndarray:
        """
        Create orange mask handling red wrap-around at hue 0/180.
        Combines three ranges: lower red (0-10), orange (5-25), upper red (170-180).
        """
        # Lower red range (0-10)
        lower_red_lower = np.array([0, 100, 100])
        lower_red_upper = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_frame, lower_red_lower, lower_red_upper)

        # Orange range (5-25)
        mask2 = cv2.inRange(hsv_frame, self.orange_lower, self.orange_upper)

        # Upper red range (170-180)
        upper_red_lower = np.array([170, 100, 100])
        upper_red_upper = np.array([180, 255, 255])
        mask3 = cv2.inRange(hsv_frame, upper_red_lower, upper_red_upper)

        # Combine all three ranges
        return cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)

    def create_white_mask(self, hsv_frame: np.ndarray) -> np.ndarray:
        """Create white mask (high brightness, low saturation)."""
        return cv2.inRange(hsv_frame, self.white_lower, self.white_upper)

    def create_combined_mask(self, hsv_frame: np.ndarray) -> np.ndarray:
        """Create combined mask of orange OR white pixels."""
        orange_mask = self.create_orange_mask(hsv_frame)
        white_mask = self.create_white_mask(hsv_frame)
        return cv2.bitwise_or(orange_mask, white_mask)


class PixelVotingFusion:
    """Validates circle candidates using pixel voting with color masks."""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def calculate_confidence(self, color_mask: np.ndarray, circle: Tuple[int, int, int]) -> float:
        """
        Calculate confidence score using pixel voting.
        Confidence = (color pixels in ROI) / (circle area)
        """
        x, y, r = circle

        # Create ROI mask (white circle on black background)
        roi_mask = np.zeros(color_mask.shape, dtype=np.uint8)
        cv2.circle(roi_mask, (x, y), r, 255, -1)

        # AND with color mask to get colored pixels within circle
        intersection = cv2.bitwise_and(color_mask, roi_mask)

        # Calculate confidence
        color_pixels = np.count_nonzero(intersection)
        circle_area = np.pi * r * r
        confidence = color_pixels / circle_area if circle_area > 0 else 0.0

        return confidence

    def validate_candidate(self, confidence: float) -> bool:
        """Return True if confidence exceeds threshold."""
        return confidence > self.threshold


class HybridBallDetector:
    """
    Hybrid ball detector using Propose & Validate approach.
    Combines Hough circles (geometry) with color validation (pixel voting).
    """

    def __init__(self):
        self.color_validator = ColorValidator()
        self.pixel_voting = PixelVotingFusion()

        # Hough circle parameters
        self.hough_param1 = 50
        self.hough_param2 = 30
        self.hough_min_radius = 10
        self.hough_max_radius = 150

        # Confidence threshold
        self.confidence_threshold = 0.6

    def find_circle_candidates(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Find circular candidates using HoughCircles.
        Returns list of (x, y, radius) tuples.
        """
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
            return [(int(x), int(y), int(r)) for x, y, r in circles[0]]
        return []

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Detect ball using hybrid Propose & Validate approach.

        Returns:
            (x, y, radius) of best validated candidate, or None if no valid candidate found
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Step 1: Create color mask
        color_mask = self.color_validator.create_combined_mask(hsv)

        # Step 2: Find circle candidates
        candidates = self.find_circle_candidates(frame)

        if not candidates:
            return None

        # Step 3: Validate candidates and find best one
        best_candidate = None
        best_confidence = 0.0

        for circle in candidates:
            confidence = self.pixel_voting.calculate_confidence(color_mask, circle)

            if confidence > self.confidence_threshold and confidence > best_confidence:
                best_confidence = confidence
                best_candidate = circle

        return best_candidate

    def detect_with_details(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int]], List[Dict]]:
        """
        Detect ball and return detailed information about all candidates.

        Returns:
            (best_candidate, all_results) where all_results contains confidence scores for all candidates
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Step 1: Create color mask
        color_mask = self.color_validator.create_combined_mask(hsv)

        # Step 2: Find circle candidates
        candidates = self.find_circle_candidates(frame)

        if not candidates:
            return None, []

        # Step 3: Validate all candidates
        results = []
        best_candidate = None
        best_confidence = 0.0

        for circle in candidates:
            confidence = self.pixel_voting.calculate_confidence(color_mask, circle)
            is_valid = confidence > self.confidence_threshold

            results.append({
                'circle': circle,
                'confidence': confidence,
                'valid': is_valid
            })

            if is_valid and confidence > best_confidence:
                best_confidence = confidence
                best_candidate = circle

        return best_candidate, results
