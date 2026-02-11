#!/usr/bin/env python3
"""
Hybrid Ball Tracker - Propose & Validate Approach
Combines Hough Circle Transform (geometry) with HSV color masking (validation)
Uses pixel voting to calculate confidence scores for each candidate.
"""

import cv2
import numpy as np


class ColorValidator:
    """Creates HSV color masks for orange and white with red wrap-around handling."""

    def __init__(self):
        # Orange HSV ranges (default values)
        self.orange_lower = np.array([5, 100, 100])
        self.orange_upper = np.array([25, 255, 255])

        # White HSV ranges
        self.white_lower = np.array([0, 0, 180])
        self.white_upper = np.array([180, 50, 255])

    def create_orange_mask(self, hsv_frame):
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

    def create_white_mask(self, hsv_frame):
        """Create white mask (high brightness, low saturation)."""
        return cv2.inRange(hsv_frame, self.white_lower, self.white_upper)

    def create_combined_mask(self, hsv_frame):
        """Create combined mask of orange OR white pixels."""
        orange_mask = self.create_orange_mask(hsv_frame)
        white_mask = self.create_white_mask(hsv_frame)
        return cv2.bitwise_or(orange_mask, white_mask)


class GeometryProposer:
    """Finds circular candidates using Hough Circle Transform."""

    def __init__(self, param1=50, param2=30, min_radius=10, max_radius=150):
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius

    def find_circle_candidates(self, frame):
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
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            return [(int(x), int(y), int(r)) for x, y, r in circles[0]]
        return []


class PixelVotingFusion:
    """Validates circle candidates using pixel voting with color masks."""

    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def calculate_confidence(self, color_mask, circle):
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

    def validate_candidate(self, confidence):
        """Return True if confidence exceeds threshold."""
        return confidence > self.threshold


class HybridBallTracker:
    """Main orchestrator for hybrid ball tracking with real-time parameter controls."""

    def __init__(self):
        self.color_validator = ColorValidator()
        self.geometry_proposer = GeometryProposer()
        self.pixel_voting = PixelVotingFusion()

        # Trackbar values
        self.params = {
            'orange_h_min': 5,
            'orange_h_max': 25,
            'orange_s_min': 100,
            'orange_v_min': 100,
            'white_v_min': 180,
            'white_s_max': 50,
            'hough_param2': 30,
            'confidence_threshold': 60  # 0.6 * 100 for trackbar
        }

        self.setup_trackbars()

    def setup_trackbars(self):
        """Create OpenCV trackbars for real-time parameter adjustment."""
        cv2.namedWindow('Hybrid Ball Tracker')
        cv2.namedWindow('Controls')

        # Orange HSV trackbars
        cv2.createTrackbar('Orange H Min', 'Controls', 5, 180, self.on_trackbar)
        cv2.createTrackbar('Orange H Max', 'Controls', 25, 180, self.on_trackbar)
        cv2.createTrackbar('Orange S Min', 'Controls', 100, 255, self.on_trackbar)
        cv2.createTrackbar('Orange V Min', 'Controls', 100, 255, self.on_trackbar)

        # White HSV trackbars
        cv2.createTrackbar('White V Min', 'Controls', 180, 255, self.on_trackbar)
        cv2.createTrackbar('White S Max', 'Controls', 50, 255, self.on_trackbar)

        # Hough parameters
        cv2.createTrackbar('Hough Param2', 'Controls', 30, 100, self.on_trackbar)

        # Confidence threshold (0-100 representing 0.0-1.0)
        cv2.createTrackbar('Confidence x100', 'Controls', 60, 100, self.on_trackbar)

    def on_trackbar(self, val):
        """Callback for trackbar changes - updates parameters."""
        self.params['orange_h_min'] = cv2.getTrackbarPos('Orange H Min', 'Controls')
        self.params['orange_h_max'] = cv2.getTrackbarPos('Orange H Max', 'Controls')
        self.params['orange_s_min'] = cv2.getTrackbarPos('Orange S Min', 'Controls')
        self.params['orange_v_min'] = cv2.getTrackbarPos('Orange V Min', 'Controls')
        self.params['white_v_min'] = cv2.getTrackbarPos('White V Min', 'Controls')
        self.params['white_s_max'] = cv2.getTrackbarPos('White S Max', 'Controls')
        self.params['hough_param2'] = cv2.getTrackbarPos('Hough Param2', 'Controls')
        self.params['confidence_threshold'] = cv2.getTrackbarPos('Confidence x100', 'Controls')

        # Update component parameters
        self.update_parameters()

    def update_parameters(self):
        """Apply trackbar values to detection components."""
        self.color_validator.orange_lower = np.array([
            self.params['orange_h_min'],
            self.params['orange_s_min'],
            self.params['orange_v_min']
        ])
        self.color_validator.orange_upper = np.array([
            self.params['orange_h_max'],
            255,
            255
        ])
        self.color_validator.white_lower = np.array([
            0,
            0,
            self.params['white_v_min']
        ])
        self.color_validator.white_upper = np.array([
            180,
            self.params['white_s_max'],
            255
        ])

        self.geometry_proposer.param2 = self.params['hough_param2']
        self.pixel_voting.threshold = self.params['confidence_threshold'] / 100.0

    def process_frame(self, frame):
        """
        Main processing pipeline: Propose & Validate.
        Returns list of results and color mask.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Step 1: Color Validator - create combined mask
        color_mask = self.color_validator.create_combined_mask(hsv)

        # Step 2: Geometry Proposer - find circle candidates
        candidates = self.geometry_proposer.find_circle_candidates(frame)

        # Step 3: Pixel Voting Fusion - validate each candidate
        results = []
        for circle in candidates:
            confidence = self.pixel_voting.calculate_confidence(color_mask, circle)
            is_valid = self.pixel_voting.validate_candidate(confidence)
            results.append({
                'circle': circle,
                'confidence': confidence,
                'valid': is_valid
            })

        return results, color_mask

    def display_results(self, frame, results):
        """
        Display results with color-coded circles and confidence scores.
        Green = confirmed (>threshold), Red = rejected (<threshold)
        """
        display_frame = frame.copy()

        for result in results:
            x, y, r = result['circle']
            confidence = result['confidence']
            is_valid = result['valid']

            # Color: green if confirmed (>0.6), red if rejected (<0.6)
            color = (0, 255, 0) if is_valid else (0, 0, 255)

            # Draw circle
            cv2.circle(display_frame, (x, y), r, color, 2)
            cv2.circle(display_frame, (x, y), 2, color, -1)

            # Display confidence score next to circle
            text = f"{confidence:.2f}"
            cv2.putText(
                display_frame,
                text,
                (x + r + 5, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return display_frame

    def run(self):
        """Main loop - capture from webcam and process frames."""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("=" * 60)
        print("Hybrid Ball Tracker Started")
        print("=" * 60)
        print("Controls:")
        print("  - Adjust trackbars in 'Controls' window to tune parameters")
        print("  - Press 'q' to quit")
        print("  - Press 'm' to toggle mask view")
        print("  - Press 'h' to show/hide help")
        print("=" * 60)

        show_mask = False
        show_help = True

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break

            # Process frame
            results, color_mask = self.process_frame(frame)

            # Display results
            display_frame = self.display_results(frame, results)

            # Add info overlay
            confirmed = sum(1 for r in results if r['valid'])
            rejected = len(results) - confirmed

            info_text = f"Candidates: {len(results)} | Confirmed: {confirmed} | Rejected: {rejected}"
            cv2.putText(
                display_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Show help text
            if show_help:
                help_lines = [
                    "GREEN = Confirmed (>threshold)",
                    "RED = Rejected (<threshold)",
                    "Press 'h' to hide help"
                ]
                y_offset = 60
                for line in help_lines:
                    cv2.putText(
                        display_frame,
                        line,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
                    y_offset += 25

            # Show frames
            cv2.imshow('Hybrid Ball Tracker', display_frame)

            if show_mask:
                cv2.imshow('Color Mask', color_mask)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow('Color Mask')
            elif key == ord('h'):
                show_help = not show_help

        cap.release()
        cv2.destroyAllWindows()
        print("\nHybrid Ball Tracker stopped")


def main():
    """Entry point."""
    tracker = HybridBallTracker()
    tracker.run()


if __name__ == "__main__":
    main()
