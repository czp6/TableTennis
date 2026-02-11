#!/usr/bin/env python3
"""
YOLOv8-based ball detector for table tennis ball tracking.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import json


def prepare_yolo_dataset(labels_file: str, output_dir: str = "yolo_dataset"):
    """
    Convert labeled data to YOLO format for training.

    Args:
        labels_file: Path to labels JSON file
        output_dir: Output directory for YOLO dataset
    """
    output_path = Path(output_dir)
    images_dir = output_path / "images" / "train"
    labels_dir = output_path / "labels" / "train"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    with open(labels_file, 'r') as f:
        data = json.load(f)

    video_path = data['video_path']
    labels = data['labels']

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    print(f"Preparing YOLO dataset from {len(labels)} labeled frames...")

    for frame_idx_str, label in labels.items():
        frame_idx = int(frame_idx_str)

        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Rotate frame
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        height, width = frame.shape[:2]

        # Save image
        image_filename = f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(images_dir / image_filename), frame)

        # Create YOLO annotation (class x_center y_center width height - normalized)
        x, y, r = label['x'], label['y'], label['radius']
        x_center = x / width
        y_center = y / height
        bbox_width = (r * 2) / width
        bbox_height = (r * 2) / height

        # Write annotation
        annotation_filename = f"frame_{frame_idx:06d}.txt"
        with open(labels_dir / annotation_filename, 'w') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

    cap.release()

    # Create data.yaml
    yaml_content = f"""
path: {output_path.absolute()}
train: images/train
val: images/train  # Using same for validation (small dataset)

nc: 1  # number of classes
names: ['ball']  # class names
"""

    with open(output_path / "data.yaml", 'w') as f:
        f.write(yaml_content)

    print(f"Dataset prepared in {output_path}")
    print(f"Images: {len(list(images_dir.glob('*.jpg')))}")
    print(f"Labels: {len(list(labels_dir.glob('*.txt')))}")
    print(f"\nTo train YOLOv8:")
    print(f"  yolo task=detect mode=train model=yolov8n.pt data={output_path}/data.yaml epochs=100 imgsz=640")


def train_yolo_detector(data_yaml: str = "yolo_dataset/data.yaml", epochs: int = 100, patience: int = 50):
    """
    Train YOLOv8 detector on labeled data.

    Args:
        data_yaml: Path to data.yaml file
        epochs: Number of training epochs
        patience: Early stopping patience (higher = less early stopping)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Install with: pip install ultralytics")
        return

    print(f"Training YOLOv8 detector (patience={patience})...")

    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=8,
        name='ball_detector',
        patience=patience,
        save=True,
        device='mps' if cv2.ocl.haveOpenCL() else 'cpu'  # Use Mac GPU if available
    )

    print(f"\nTraining complete! Model saved to: runs/detect/ball_detector/weights/best.pt")
    return results


class YOLOBallDetector:
    """YOLOv8-based ball detector with segmentation support."""

    def __init__(self, model_path: str = "colab_models/best.pt"):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to trained YOLO model
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.available = True
            self.last_mask = None  # Store last segmentation mask
            print(f"Loaded YOLOv8 model from {model_path}")
        except ImportError:
            print("Warning: ultralytics not installed. Install with: pip install ultralytics")
            self.available = False
            self.last_mask = None
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            self.available = False
            self.last_mask = None

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.2, min_mask_area: int = 100, radius_scale: float = 0.8, debug: bool = False) -> Optional[Tuple[int, int, int]]:
        """
        Detect ball in frame using YOLO segmentation.

        Args:
            frame: Input frame
            conf_threshold: Confidence threshold (default 0.2 for optimal balance)
            min_mask_area: Minimum mask area in pixels to accept detection
            radius_scale: Scale factor for circle radius (0.0-1.0)
            debug: Print debug information to console

        Returns:
            (x, y, radius) tuple or None
        """
        if not self.available:
            return None

        # Run inference
        results = self.model(frame, conf=conf_threshold, verbose=False)

        # Get detections
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Get the detection with highest confidence
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)
            best_conf = confidences[best_idx]

            if debug:
                print(f"[DEBUG] YOLOv8 Detection - Confidence: {best_conf:.3f}")

            # Get bounding box
            box = boxes.xyxy[best_idx].cpu().numpy()
            x1, y1, x2, y2 = box

            # Calculate center and radius using circumscribed circle
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            width = x2 - x1
            height = y2 - y1
            radius = int(np.sqrt(width**2 + height**2) / 2)

            # Store segmentation mask if available
            if hasattr(results[0], 'masks') and results[0].masks is not None:
                mask = results[0].masks.data[best_idx].cpu().numpy()
                # Resize mask to frame size
                import cv2
                self.last_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                # Check mask area - MUST have valid mask to proceed
                mask_area = np.sum(self.last_mask > 0.5)

                if debug:
                    print(f"[DEBUG] Segmentation mask pixels: {mask_area}")
                    print(f"[DEBUG] Min mask area threshold: {min_mask_area}")
                    print(f"[DEBUG] Original radius: {radius}, Scaled radius: {int(radius * radius_scale)}")

                if mask_area < min_mask_area:
                    # Mask too small, likely false positive
                    if debug:
                        print(f"[DEBUG] REJECTED - Mask area {mask_area} < threshold {min_mask_area}")
                    self.last_mask = None
                    return None

                # Apply radius scaling
                radius = int(radius * radius_scale)

                if debug:
                    print(f"[DEBUG] ACCEPTED - Detection at ({cx}, {cy}) with radius {radius}")

                return (cx, cy, radius)
            else:
                # No segmentation mask available - reject detection
                if debug:
                    print(f"[DEBUG] REJECTED - No segmentation mask available")
                self.last_mask = None
                return None

        if debug:
            print(f"[DEBUG] No detections found")

        self.last_mask = None
        return None

    def get_last_mask(self) -> Optional[np.ndarray]:
        """
        Get the last segmentation mask.

        Returns:
            Segmentation mask or None
        """
        return self.last_mask


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python yolo_detector.py prepare <labels_file.json>")
        print("  python yolo_detector.py train [data.yaml] [epochs]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "prepare":
        if len(sys.argv) < 3:
            print("Error: labels file required")
            sys.exit(1)
        prepare_yolo_dataset(sys.argv[2])

    elif command == "train":
        data_yaml = sys.argv[2] if len(sys.argv) > 2 else "yolo_dataset/data.yaml"
        epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        patience = int(sys.argv[4]) if len(sys.argv) > 4 else 50
        train_yolo_detector(data_yaml, epochs, patience)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
