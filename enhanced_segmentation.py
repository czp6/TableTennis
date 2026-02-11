#!/usr/bin/env python3
"""
Enhanced color segmentation for orange/white ball detection.
Uses contrast enhancement and clever boundary detection techniques.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class EnhancedColorSegmentation:
    """Enhanced color segmentation with contrast enhancement and boundary detection."""

    def __init__(self):
        """Initialize color segmentation."""
        # HSV ranges for orange and white
        self.orange_lower = np.array([5, 100, 100])
        self.orange_upper = np.array([25, 255, 255])
        self.white_lower = np.array([0, 0, 180])
        self.white_upper = np.array([180, 50, 255])

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: Input BGR image

        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split channels
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # Merge channels
        lab_enhanced = cv2.merge([l_enhanced, a, b])

        # Convert back to BGR
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    def find_color_boundary(self, orange_mask: np.ndarray, white_mask: np.ndarray, ball_region: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the curved boundary arc between orange and white regions with smoothing.

        Args:
            orange_mask: Binary mask of orange region
            white_mask: Binary mask of white region
            ball_region: The ball region image for edge detection

        Returns:
            Smoothed boundary contour as numpy array or None
        """
        # Dilate both masks to create overlap region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        orange_dilated = cv2.dilate(orange_mask, kernel, iterations=3)
        white_dilated = cv2.dilate(white_mask, kernel, iterations=3)

        # Find the overlap region (this is the boundary zone)
        boundary_zone = cv2.bitwise_and(orange_dilated, white_dilated)

        if boundary_zone.sum() == 0:
            return None

        # Find the skeleton/centerline of the boundary zone
        # This gives us the middle line between orange and white
        boundary_thinned = cv2.ximgproc.thinning(boundary_zone) if hasattr(cv2, 'ximgproc') else boundary_zone

        # Extract boundary points
        boundary_points = np.column_stack(np.where(boundary_thinned > 0))

        if len(boundary_points) < 5:
            # Fallback: just use the boundary zone contour
            contours, _ = cv2.findContours(boundary_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                return None
            boundary_points = max(contours, key=cv2.contourArea).reshape(-1, 2)
            # Swap x,y since findContours returns (x,y) but np.where returns (y,x)
            boundary_points = boundary_points[:, [1, 0]]

        if len(boundary_points) < 5:
            return None

        # Swap y,x to x,y for proper coordinates
        boundary_points = boundary_points[:, [1, 0]].astype(np.float32)

        # Sort points to form a continuous curve
        # Find the centroid
        centroid = boundary_points.mean(axis=0)

        # Sort by angle from centroid
        angles = np.arctan2(boundary_points[:, 1] - centroid[1],
                           boundary_points[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        boundary_points = boundary_points[sorted_indices]

        # Smooth the boundary curve using spline interpolation
        try:
            from scipy.interpolate import splprep, splev

            # Fit a B-spline to the boundary points
            tck, u = splprep([boundary_points[:, 0], boundary_points[:, 1]],
                            s=len(boundary_points) * 3.0,  # Smoothing factor
                            per=False,  # Not periodic since it's an arc
                            k=min(3, len(boundary_points) - 1))  # Cubic spline or lower

            # Generate smooth curve with more points
            u_new = np.linspace(0, 1, 100)  # 100 points for smooth curve
            smooth_x, smooth_y = splev(u_new, tck)

            # Convert back to contour format
            smooth_boundary = np.column_stack([smooth_x, smooth_y]).astype(np.int32)
            smooth_boundary = smooth_boundary.reshape(-1, 1, 2)

            return smooth_boundary

        except (ImportError, Exception):
            # Fallback: Use simple moving average smoothing
            window_size = min(5, len(boundary_points) // 3)
            if window_size < 2:
                window_size = 2

            smoothed_x = np.convolve(boundary_points[:, 0],
                                    np.ones(window_size)/window_size,
                                    mode='same')
            smoothed_y = np.convolve(boundary_points[:, 1],
                                    np.ones(window_size)/window_size,
                                    mode='same')

            smooth_boundary = np.column_stack([smoothed_x, smoothed_y]).astype(np.int32)
            smooth_boundary = smooth_boundary.reshape(-1, 1, 2)

            return smooth_boundary

    def segment_orange_white(self, frame: np.ndarray, ball_pos: Tuple[int, int, int],
                            monte_carlo_samples: int = 1000) -> Tuple[
        Optional[Tuple[int, int]], Optional[Tuple[int, int]], np.ndarray
    ]:
        """
        Segment orange and white regions using simplified pixel-based approach.
        Brightens detected pixels and calculates centroids directly.

        Args:
            frame: Input frame
            ball_pos: (x, y, radius) of ball
            monte_carlo_samples: Number of random samples for efficiency (0 = use all pixels)

        Returns:
            (orange_centroid, white_centroid, visualization_frame)
        """
        x, y, r = ball_pos

        # Extract ball region with margin
        margin = int(r * 0.2)
        x1 = max(0, x - r - margin)
        y1 = max(0, y - r - margin)
        x2 = min(frame.shape[1], x + r + margin)
        y2 = min(frame.shape[0], y + r + margin)

        ball_region = frame[y1:y2, x1:x2].copy()

        if ball_region.size == 0:
            return None, None, frame

        # Enhance contrast
        enhanced = self.enhance_contrast(ball_region)

        # Convert to HSV
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

        # Create circular mask for ball region
        mask = np.zeros(ball_region.shape[:2], dtype=np.uint8)
        center_in_region = (r + margin, r + margin)
        cv2.circle(mask, center_in_region, r, 255, -1)

        # Detect orange pixels
        orange_mask = cv2.inRange(hsv, self.orange_lower, self.orange_upper)
        orange_mask = cv2.bitwise_and(orange_mask, orange_mask, mask=mask)

        # Detect white pixels
        white_color_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        white_mask = cv2.bitwise_and(mask, white_color_mask)

        # Get pixel coordinates
        orange_pixels = np.column_stack(np.where(orange_mask > 0))
        white_pixels = np.column_stack(np.where(white_mask > 0))

        # Monte Carlo sampling for efficiency
        if monte_carlo_samples > 0:
            if len(orange_pixels) > monte_carlo_samples:
                indices = np.random.choice(len(orange_pixels), monte_carlo_samples, replace=False)
                orange_pixels_sampled = orange_pixels[indices]
            else:
                orange_pixels_sampled = orange_pixels

            if len(white_pixels) > monte_carlo_samples:
                indices = np.random.choice(len(white_pixels), monte_carlo_samples, replace=False)
                white_pixels_sampled = white_pixels[indices]
            else:
                white_pixels_sampled = white_pixels
        else:
            orange_pixels_sampled = orange_pixels
            white_pixels_sampled = white_pixels

        # Calculate centroids
        orange_centroid = None
        white_centroid = None

        if len(orange_pixels_sampled) > 0:
            # Centroid in local coordinates (y, x from np.where)
            cy_local = int(np.mean(orange_pixels_sampled[:, 0]))
            cx_local = int(np.mean(orange_pixels_sampled[:, 1]))
            # Convert to full frame coordinates
            orange_centroid = (cx_local + x1, cy_local + y1)

        if len(white_pixels_sampled) > 0:
            cy_local = int(np.mean(white_pixels_sampled[:, 0]))
            cx_local = int(np.mean(white_pixels_sampled[:, 1]))
            white_centroid = (cx_local + x1, cy_local + y1)

        # Create visualization frame with brightened pixels
        vis_frame = frame.copy()

        # Brighten orange pixels (increase brightness by 50%)
        for py, px in orange_pixels:
            fy, fx = py + y1, px + x1
            if 0 <= fy < vis_frame.shape[0] and 0 <= fx < vis_frame.shape[1]:
                vis_frame[fy, fx] = np.clip(vis_frame[fy, fx] * 1.5, 0, 255).astype(np.uint8)

        # Brighten white pixels (increase brightness by 50%)
        for py, px in white_pixels:
            fy, fx = py + y1, px + x1
            if 0 <= fy < vis_frame.shape[0] and 0 <= fx < vis_frame.shape[1]:
                vis_frame[fy, fx] = np.clip(vis_frame[fy, fx] * 1.5, 0, 255).astype(np.uint8)

        return orange_centroid, white_centroid, vis_frame

    def calculate_rotation_angle(self, ball_center: Tuple[int, int], orange_centroid: Optional[Tuple[int, int]],
                                 white_centroid: Optional[Tuple[int, int]]) -> Optional[float]:
        """
        Calculate rotation angle from ball center to orange centroid.

        Args:
            ball_center: (x, y) of ball center
            orange_centroid: (x, y) of orange region centroid
            white_centroid: (x, y) of white region centroid

        Returns:
            Angle in degrees or None
        """
        if orange_centroid is None:
            return None

        bx, by = ball_center
        ox, oy = orange_centroid

        # Calculate angle from ball center to orange centroid
        angle = np.arctan2(oy - by, ox - bx) * 180 / np.pi

        # Normalize to 0-360
        if angle < 0:
            angle += 360

        return angle


def test_enhanced_segmentation(video_path: str, frame_idx: int = 54):
    """Test enhanced color segmentation on a specific frame."""
    segmenter = EnhancedColorSegmentation()

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    if not ret:
        print(f"Could not read frame {frame_idx}")
        return

    # Rotate frame
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Known ball position from labels (frame 54)
    ball_pos = (998, 576, 66)

    # Segment colors
    orange_cent, white_cent, orange_cont, white_cont = segmenter.segment_orange_white(frame, ball_pos)

    # Draw results
    x, y, r = ball_pos
    cv2.circle(frame, (x, y), r, (0, 255, 0), 2)

    if orange_cent:
        cv2.circle(frame, orange_cent, 8, (0, 165, 255), -1)
        cv2.line(frame, (x, y), orange_cent, (0, 165, 255), 2)

    if white_cent:
        cv2.circle(frame, white_cent, 8, (255, 255, 255), -1)
        cv2.line(frame, (x, y), white_cent, (200, 200, 200), 2)

    if orange_cont is not None:
        cv2.drawContours(frame, [orange_cont], -1, (0, 165, 255), 2)

    if white_cont is not None:
        cv2.drawContours(frame, [white_cont], -1, (255, 255, 255), 2)

    # Calculate angle
    if orange_cent:
        angle = segmenter.calculate_rotation_angle((x, y), orange_cent, white_cent)
        print(f"Rotation angle: {angle:.2f}°")

    # Display
    cv2.imshow('Enhanced Segmentation', cv2.resize(frame, (800, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cap.release()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python enhanced_segmentation.py <video_path> [frame_idx]")
        sys.exit(1)

    video_path = sys.argv[1]
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 54

    test_enhanced_segmentation(video_path, frame_idx)
