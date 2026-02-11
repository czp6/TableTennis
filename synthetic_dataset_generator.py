#!/usr/bin/env python3
"""
Synthetic dataset generator for half-orange, half-white table tennis balls.
Generates images with segmentation masks in YOLO format.
"""

import cv2
import numpy as np
from pathlib import Path
import random
import json


class SyntheticBallGenerator:
    """Generate synthetic images of half-orange, half-white balls with segmentation masks."""

    def __init__(self, image_size=(640, 480)):
        self.image_size = image_size
        self.width, self.height = image_size

    def generate_background(self, bg_type='table'):
        """Generate various background types."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if bg_type == 'table':
            # Green table tennis table
            color = (random.randint(20, 60), random.randint(80, 140), random.randint(20, 60))
            img[:] = color
            # Add table lines
            cv2.line(img, (self.width//2, 0), (self.width//2, self.height), (255, 255, 255), 2)

        elif bg_type == 'floor':
            # Wooden floor
            color = (random.randint(100, 150), random.randint(120, 170), random.randint(140, 200))
            img[:] = color

        elif bg_type == 'court':
            # Blue court
            color = (random.randint(100, 180), random.randint(50, 100), random.randint(20, 60))
            img[:] = color

        elif bg_type == 'plain':
            # Plain background
            color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            img[:] = color

        # Add noise
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    def draw_half_ball(self, img, center, radius, rotation_angle):
        """
        Draw a half-orange, half-white ball and return segmentation mask.

        Returns:
            mask: Binary mask of the ball
            polygon: List of (x, y) points defining the ball contour
        """
        x, y = center

        # Create mask for the ball
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)

        # Create temporary image for the ball
        ball_img = img.copy()

        # Draw white half
        white_color = (255, 255, 255)
        cv2.circle(ball_img, (x, y), radius, white_color, -1)

        # Draw orange half (rotated)
        orange_color = (0, 140, 255)  # BGR format

        # Create a mask for the orange half
        orange_mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # Calculate the dividing line based on rotation
        angle_rad = np.radians(rotation_angle)

        # Create points for a half-circle
        points = []
        for angle in np.linspace(angle_rad - np.pi/2, angle_rad + np.pi/2, 100):
            px = int(x + radius * 1.5 * np.cos(angle))
            py = int(y + radius * 1.5 * np.sin(angle))
            points.append([px, py])

        # Close the polygon through center
        points.append([x, y])
        points = np.array(points, dtype=np.int32)

        cv2.fillPoly(orange_mask, [points], 255)

        # Apply orange color where mask is set
        ball_img[orange_mask > 0] = orange_color

        # Blend the ball onto the original image
        img[mask > 0] = ball_img[mask > 0]

        # Add some shading for realism
        overlay = img.copy()
        cv2.circle(overlay, (x + radius//4, y - radius//4), radius//3, (255, 255, 255), -1)
        img = cv2.addWeighted(img, 0.9, overlay, 0.1, 0)

        # Generate polygon points for YOLO segmentation format (circle approximation)
        polygon = []
        num_points = 36  # Number of points to approximate circle
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            px = x + radius * np.cos(angle)
            py = y + radius * np.sin(angle)
            # Normalize to [0, 1]
            polygon.append((px / self.width, py / self.height))

        return img, mask, polygon

    def add_distractors(self, img):
        """Add distractor objects that might confuse the detector."""
        distractor_type = random.choice(['circle', 'ellipse', 'rectangle', 'none'])

        if distractor_type == 'circle':
            # Add a circular object with different color
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
        """
        Generate a single training sample.

        Returns:
            img: Generated image
            polygon: Segmentation polygon (None if no ball)
        """
        # Generate background
        bg_type = random.choice(['table', 'floor', 'court', 'plain'])
        img = self.generate_background(bg_type)

        polygon = None

        if include_ball:
            # Random ball parameters
            radius = random.randint(15, 60)
            x = random.randint(radius + 10, self.width - radius - 10)
            y = random.randint(radius + 10, self.height - radius - 10)
            rotation = random.randint(0, 360)

            # Draw ball
            img, mask, polygon = self.draw_half_ball(img, (x, y), radius, rotation)

        if include_distractors:
            img = self.add_distractors(img)

        # Add blur for realism
        if random.random() > 0.5:
            kernel_size = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        # Add brightness/contrast variation
        alpha = random.uniform(0.7, 1.3)  # Contrast
        beta = random.randint(-30, 30)    # Brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        return img, polygon

    def save_yolo_segmentation(self, polygon, output_path):
        """Save segmentation polygon in YOLO format."""
        if polygon is None:
            # Empty file for negative samples
            with open(output_path, 'w') as f:
                pass
            return

        # YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 ...
        with open(output_path, 'w') as f:
            # Class 0 for ball
            line = "0"
            for x, y in polygon:
                line += f" {x:.6f} {y:.6f}"
            f.write(line + "\n")


def generate_dataset(output_dir, num_positive=500, num_negative=100, num_distractor=100):
    """
    Generate complete synthetic dataset.

    Args:
        output_dir: Output directory path
        num_positive: Number of images with balls
        num_negative: Number of images without balls
        num_distractor: Number of images with balls and distractors
    """
    output_path = Path(output_dir)

    # Create directory structure
    images_train = output_path / "images" / "train"
    images_val = output_path / "images" / "val"
    labels_train = output_path / "labels" / "train"
    labels_val = output_path / "labels" / "val"

    for dir_path in [images_train, images_val, labels_train, labels_val]:
        dir_path.mkdir(parents=True, exist_ok=True)

    generator = SyntheticBallGenerator()

    print(f"Generating {num_positive} positive samples...")
    for i in range(num_positive):
        img, polygon = generator.generate_sample(include_ball=True, include_distractors=False)

        # 80/20 train/val split
        if i < num_positive * 0.8:
            img_path = images_train / f"positive_{i:04d}.jpg"
            label_path = labels_train / f"positive_{i:04d}.txt"
        else:
            img_path = images_val / f"positive_{i:04d}.jpg"
            label_path = labels_val / f"positive_{i:04d}.txt"

        cv2.imwrite(str(img_path), img)
        generator.save_yolo_segmentation(polygon, label_path)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_positive} positive samples")

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
            print(f"  Generated {i + 1}/{num_negative} negative samples")

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
            print(f"  Generated {i + 1}/{num_distractor} distractor samples")

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
        'total_samples': num_positive + num_negative + num_distractor,
        'positive_samples': num_positive,
        'negative_samples': num_negative,
        'distractor_samples': num_distractor,
        'train_samples': int((num_positive + num_negative + num_distractor) * 0.8),
        'val_samples': int((num_positive + num_negative + num_distractor) * 0.2),
        'image_size': [640, 480],
        'classes': ['ball']
    }

    with open(output_path / "dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Dataset generation complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    print(f"Total samples: {info['total_samples']}")
    print(f"  - Positive (ball only): {num_positive}")
    print(f"  - Negative (no ball): {num_negative}")
    print(f"  - Distractor (ball + distractors): {num_distractor}")
    print(f"Train/Val split: {info['train_samples']}/{info['val_samples']}")
    print(f"\nData config: {output_path / 'data.yaml'}")
    print(f"Dataset info: {output_path / 'dataset_info.json'}")


if __name__ == "__main__":
    import sys

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "synthetic_ball_dataset"
    num_positive = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    num_negative = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    num_distractor = int(sys.argv[4]) if len(sys.argv) > 4 else 100

    print("Synthetic Ball Dataset Generator")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Positive samples: {num_positive}")
    print(f"Negative samples: {num_negative}")
    print(f"Distractor samples: {num_distractor}")
    print("="*60)

    generate_dataset(output_dir, num_positive, num_negative, num_distractor)
