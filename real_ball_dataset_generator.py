#!/usr/bin/env python3
"""
Real ball dataset generator using cropped table tennis ball photos.
Extracts real ball images and composites them onto various backgrounds.
"""

import cv2
import numpy as np
from pathlib import Path
import random
import json
from typing import List, Tuple, Optional


class RealBallExtractor:
    """Extract ball crops from video frames or images."""

    def __init__(self, video_path: str, labels_file: Optional[str] = None):
        """
        Initialize extractor.

        Args:
            video_path: Path to video file
            labels_file: Optional JSON file with labeled ball positions
        """
        self.video_path = video_path
        self.labels_file = labels_file
        self.ball_crops = []

    def extract_from_labels(self, max_crops: int = 50) -> List[np.ndarray]:
        """
        Extract ball crops from labeled frames.

        Args:
            max_crops: Maximum number of ball crops to extract

        Returns:
            List of cropped ball images
        """
        if not self.labels_file or not Path(self.labels_file).exists():
            print(f"Labels file not found: {self.labels_file}")
            return []

        # Load labels
        with open(self.labels_file, 'r') as f:
            data = json.load(f)

        video_path = data.get('video_path', self.video_path)
        labels = data.get('labels', {})

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return []

        print(f"Extracting ball crops from {len(labels)} labeled frames...")

        crops = []
        for frame_idx_str, label in list(labels.items())[:max_crops]:
            frame_idx = int(frame_idx_str)

            # Read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Rotate frame 90 degrees clockwise
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Extract ball region
            x, y, r = label['x'], label['y'], label['radius']

            # Add margin around ball
            margin = int(r * 1.5)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + margin)
            y2 = min(frame.shape[0], y + margin)

            # Crop ball region
            ball_crop = frame[y1:y2, x1:x2].copy()

            if ball_crop.size > 0:
                crops.append(ball_crop)

            if len(crops) % 10 == 0:
                print(f"  Extracted {len(crops)} ball crops")

        cap.release()

        print(f"Extracted {len(crops)} ball crops total")
        self.ball_crops = crops
        return crops

    def extract_manual(self, frame_indices: List[int], ball_positions: List[Tuple[int, int, int]]) -> List[np.ndarray]:
        """
        Extract ball crops from manually specified positions.

        Args:
            frame_indices: List of frame indices
            ball_positions: List of (x, y, radius) tuples

        Returns:
            List of cropped ball images
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Could not open video: {self.video_path}")
            return []

        crops = []
        for frame_idx, (x, y, r) in zip(frame_indices, ball_positions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Rotate frame 90 degrees clockwise
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Add margin
            margin = int(r * 1.5)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + margin)
            y2 = min(frame.shape[0], y + margin)

            ball_crop = frame[y1:y2, x1:x2].copy()
            if ball_crop.size > 0:
                crops.append(ball_crop)

        cap.release()
        self.ball_crops = crops
        return crops


class RealBallDatasetGenerator:
    """Generate dataset using real ball crops composited onto backgrounds."""

    def __init__(self, ball_crops: List[np.ndarray], image_size=(640, 480)):
        self.ball_crops = ball_crops
        self.image_size = image_size
        self.width, self.height = image_size

    def generate_background(self, bg_type='table'):
        """Generate various background types."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if bg_type == 'table':
            color = (random.randint(20, 60), random.randint(80, 140), random.randint(20, 60))
            img[:] = color
            cv2.line(img, (self.width//2, 0), (self.width//2, self.height), (255, 255, 255), 2)

        elif bg_type == 'floor':
            color = (random.randint(100, 150), random.randint(120, 170), random.randint(140, 200))
            img[:] = color

        elif bg_type == 'court':
            color = (random.randint(100, 180), random.randint(50, 100), random.randint(20, 60))
            img[:] = color

        elif bg_type == 'plain':
            color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            img[:] = color

        # Add noise
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    def composite_ball(self, background: np.ndarray, ball_crop: np.ndarray,
                      position: Tuple[int, int], scale: float = 1.0,
                      rotation: float = 0.0) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]]]:
        """
        Composite real ball crop onto background.

        Returns:
            img: Composited image
            mask: Binary mask of ball
            polygon: Segmentation polygon
        """
        # Resize ball crop
        if scale != 1.0:
            new_size = (int(ball_crop.shape[1] * scale), int(ball_crop.shape[0] * scale))
            ball_crop = cv2.resize(ball_crop, new_size)

        # Rotate ball crop
        if rotation != 0.0:
            center = (ball_crop.shape[1] // 2, ball_crop.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            ball_crop = cv2.warpAffine(ball_crop, matrix, (ball_crop.shape[1], ball_crop.shape[0]))

        # Calculate position
        x, y = position
        ball_h, ball_w = ball_crop.shape[:2]

        # Calculate desired position
        desired_x1 = x - ball_w // 2
        desired_y1 = y - ball_h // 2
        desired_x2 = desired_x1 + ball_w
        desired_y2 = desired_y1 + ball_h

        # Clip to image boundaries
        x1 = max(0, desired_x1)
        y1 = max(0, desired_y1)
        x2 = min(self.width, desired_x2)
        y2 = min(self.height, desired_y2)

        # Calculate corresponding crop region
        crop_x1 = x1 - desired_x1
        crop_y1 = y1 - desired_y1
        crop_x2 = crop_x1 + (x2 - x1)
        crop_y2 = crop_y1 + (y2 - y1)

        # Create mask for ball (circular)
        ball_mask = np.zeros((ball_h, ball_w), dtype=np.uint8)
        center = (ball_w // 2, ball_h // 2)
        radius = min(ball_w, ball_h) // 2
        cv2.circle(ball_mask, center, radius, 255, -1)

        # Composite ball onto background
        img = background.copy()
        roi = img[y1:y2, x1:x2]
        ball_region = ball_crop[crop_y1:crop_y2, crop_x1:crop_x2]
        mask_region = ball_mask[crop_y1:crop_y2, crop_x1:crop_x2]

        # Ensure shapes match
        if roi.shape[:2] != ball_region.shape[:2] or roi.shape[:2] != mask_region.shape:
            # Skip this sample if shapes don't match
            return background, np.zeros((self.height, self.width), dtype=np.uint8), []

        # Blend using mask
        for c in range(3):
            roi[:, :, c] = np.where(mask_region > 0, ball_region[:, :, c], roi[:, :, c])

        img[y1:y2, x1:x2] = roi

        # Create full mask
        full_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask_region

        # Generate polygon for segmentation
        polygon = []
        num_points = 36
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            px = x + radius * np.cos(angle)
            py = y + radius * np.sin(angle)
            # Normalize to [0, 1]
            polygon.append((px / self.width, py / self.height))

        return img, full_mask, polygon

    def add_distractors(self, img):
        """Add distractor objects."""
        distractor_type = random.choice(['circle', 'ellipse', 'rectangle', 'none'])

        if distractor_type == 'circle':
            x = random.randint(50, self.width - 50)
            y = random.randint(50, self.height - 50)
            r = random.randint(10, 40)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.circle(img, (x, y), r, color, -1)

        elif distractor_type == 'ellipse':
            x = random.randint(50, self.width - 50)
            y = random.randint(50, self.height - 50)
            axes = (random.randint(15, 40), random.randint(10, 30))
            angle = random.randint(0, 180)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.ellipse(img, (x, y), axes, angle, 0, 360, color, -1)

        elif distractor_type == 'rectangle':
            x1 = random.randint(20, self.width - 60)
            y1 = random.randint(20, self.height - 60)
            x2 = x1 + random.randint(20, 50)
            y2 = y1 + random.randint(20, 50)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

        return img

    def generate_sample(self, include_ball=True, include_distractors=False):
        """Generate a training sample using real ball crop."""
        bg_type = random.choice(['table', 'floor', 'court', 'plain'])
        img = self.generate_background(bg_type)

        polygon = None

        if include_ball and len(self.ball_crops) > 0:
            # Select random ball crop
            ball_crop = random.choice(self.ball_crops)

            # Random parameters
            scale = random.uniform(0.5, 1.5)
            rotation = random.randint(0, 360)
            x = random.randint(50, self.width - 50)
            y = random.randint(50, self.height - 50)

            img, mask, polygon = self.composite_ball(img, ball_crop, (x, y), scale, rotation)

        if include_distractors:
            img = self.add_distractors(img)

        # Add blur
        if random.random() > 0.5:
            kernel_size = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        # Add brightness/contrast variation
        alpha = random.uniform(0.7, 1.3)
        beta = random.randint(-30, 30)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        return img, polygon

    def save_yolo_segmentation(self, polygon, output_path):
        """Save segmentation polygon in YOLO format."""
        if polygon is None:
            with open(output_path, 'w') as f:
                pass
            return

        with open(output_path, 'w') as f:
            line = "0"
            for x, y in polygon:
                line += f" {x:.6f} {y:.6f}"
            f.write(line + "\n")


def generate_dataset_from_video(video_path: str, labels_file: str, output_dir: str,
                                num_positive=500, num_negative=100, num_distractor=100,
                                max_ball_crops=50):
    """
    Generate dataset using real ball crops from video.

    Args:
        video_path: Path to video file
        labels_file: Path to labels JSON file
        output_dir: Output directory
        num_positive: Number of positive samples
        num_negative: Number of negative samples
        num_distractor: Number of distractor samples
        max_ball_crops: Maximum ball crops to extract
    """
    output_path = Path(output_dir)

    # Create directory structure
    images_train = output_path / "images" / "train"
    images_val = output_path / "images" / "val"
    labels_train = output_path / "labels" / "train"
    labels_val = output_path / "labels" / "val"

    for dir_path in [images_train, images_val, labels_train, labels_val]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Extract ball crops
    print("="*60)
    print("Step 1: Extracting ball crops from video")
    print("="*60)
    extractor = RealBallExtractor(video_path, labels_file)
    ball_crops = extractor.extract_from_labels(max_crops=max_ball_crops)

    if len(ball_crops) == 0:
        print("Error: No ball crops extracted. Check video and labels file.")
        return

    print(f"\nExtracted {len(ball_crops)} ball crops")

    # Generate dataset
    print("\n" + "="*60)
    print("Step 2: Generating training dataset")
    print("="*60)

    generator = RealBallDatasetGenerator(ball_crops)

    print(f"\nGenerating {num_positive} positive samples...")
    for i in range(num_positive):
        img, polygon = generator.generate_sample(include_ball=True, include_distractors=False)

        if i < num_positive * 0.8:
            img_path = images_train / f"positive_{i:04d}.jpg"
            label_path = labels_train / f"positive_{i:04d}.txt"
        else:
            img_path = images_val / f"positive_{i:04d}.jpg"
            label_path = labels_val / f"positive_{i:04d}.txt"

        cv2.imwrite(str(img_path), img)
        generator.save_yolo_segmentation(polygon, label_path)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_positive}")

    print(f"\nGenerating {num_negative} negative samples...")
    for i in range(num_negative):
        img, polygon = generator.generate_sample(include_ball=False, include_distractors=True)

        if i < num_negative * 0.8:
            img_path = images_train / f"negative_{i:04d}.jpg"
            label_path = labels_train / f"negative_{i:04d}.txt"
        else:
            img_path = images_val / f"negative_{i:04d}.jpg"
            label_path = labels_val / f"negative_{i:04d}.txt"

        cv2.imwrite(str(img_path), img)
        generator.save_yolo_segmentation(polygon, label_path)

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_negative}")

    print(f"\nGenerating {num_distractor} distractor samples...")
    for i in range(num_distractor):
        img, polygon = generator.generate_sample(include_ball=True, include_distractors=True)

        if i < num_distractor * 0.8:
            img_path = images_train / f"distractor_{i:04d}.jpg"
            label_path = labels_train / f"distractor_{i:04d}.txt"
        else:
            img_path = images_val / f"distractor_{i:04d}.jpg"
            label_path = labels_val / f"distractor_{i:04d}.txt"

        cv2.imwrite(str(img_path), img)
        generator.save_yolo_segmentation(polygon, label_path)

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_distractor}")

    # Create data.yaml
    yaml_content = f"""path: {output_path.absolute()}
train: images/train
val: images/val

nc: 1
names: ['ball']
"""

    with open(output_path / "data.yaml", 'w') as f:
        f.write(yaml_content)

    # Create dataset info
    info = {
        'source_video': video_path,
        'labels_file': labels_file,
        'ball_crops_extracted': len(ball_crops),
        'total_samples': num_positive + num_negative + num_distractor,
        'positive_samples': num_positive,
        'negative_samples': num_negative,
        'distractor_samples': num_distractor,
        'train_samples': int((num_positive + num_negative + num_distractor) * 0.8),
        'val_samples': int((num_positive + num_negative + num_distractor) * 0.2),
    }

    with open(output_path / "dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Dataset generation complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    print(f"Ball crops used: {len(ball_crops)}")
    print(f"Total samples: {info['total_samples']}")
    print(f"  - Positive: {num_positive}")
    print(f"  - Negative: {num_negative}")
    print(f"  - Distractor: {num_distractor}")
    print(f"Train/Val split: {info['train_samples']}/{info['val_samples']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python real_ball_dataset_generator.py <video_path> <labels_file> [output_dir] [num_positive] [num_negative] [num_distractor]")
        print("\nExample:")
        print("  python real_ball_dataset_generator.py video.mp4 labels.json real_ball_dataset 500 100 100")
        sys.exit(1)

    video_path = sys.argv[1]
    labels_file = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "real_ball_dataset"
    num_positive = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    num_negative = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    num_distractor = int(sys.argv[6]) if len(sys.argv) > 6 else 100

    generate_dataset_from_video(
        video_path, labels_file, output_dir,
        num_positive, num_negative, num_distractor
    )
