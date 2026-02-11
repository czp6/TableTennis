#!/usr/bin/env python3
"""
Google Colab script for training YOLOv8 segmentation model.
Copy and paste cells into Google Colab notebook.

Note: YOLOv8 uses PyTorch which runs on GPU, not TPU.
Make sure to enable GPU in Colab: Runtime > Change runtime type > GPU
"""

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
CELL_1_INSTALL = """
!pip install ultralytics opencv-python-headless
!pip install --upgrade pip

# Verify GPU is available
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""

# ============================================================================
# CELL 2: Mount Google Drive (Optional - if dataset is on Drive)
# ============================================================================
CELL_2_MOUNT = """
from google.colab import drive
drive.mount('/content/drive')
"""

# ============================================================================
# CELL 3: Upload Dataset or Generate Synthetic Data
# ============================================================================
CELL_3_DATASET = """
import os
from pathlib import Path

# Option A: Upload dataset ZIP file
# Uncomment if you have a pre-generated dataset
# from google.colab import files
# uploaded = files.upload()
# !unzip -q synthetic_ball_dataset.zip -d /content/

# Option B: Generate synthetic dataset in Colab
# First, upload the synthetic_dataset_generator.py file, then:

# Upload the generator script
from google.colab import files
print("Upload synthetic_dataset_generator.py:")
uploaded = files.upload()

# Generate dataset
!python synthetic_dataset_generator.py /content/ball_dataset 500 100 100

# Verify dataset structure
dataset_path = Path("/content/ball_dataset")
print(f"\\nDataset structure:")
print(f"Train images: {len(list((dataset_path / 'images' / 'train').glob('*.jpg')))}")
print(f"Val images: {len(list((dataset_path / 'images' / 'val').glob('*.jpg')))}")
print(f"Train labels: {len(list((dataset_path / 'labels' / 'train').glob('*.txt')))}")
print(f"Val labels: {len(list((dataset_path / 'labels' / 'val').glob('*.txt')))}")
"""

# ============================================================================
# CELL 4: Train YOLOv8 Segmentation Model
# ============================================================================
CELL_4_TRAIN = """
from ultralytics import YOLO

# Load a pretrained YOLOv8 segmentation model
# Options: yolov8n-seg.pt (nano), yolov8s-seg.pt (small), yolov8m-seg.pt (medium)
model = YOLO('yolov8n-seg.pt')  # Nano model for faster training

# Train the model
results = model.train(
    data='/content/ball_dataset/data.yaml',
    epochs=100,                    # Number of training epochs
    imgsz=640,                     # Image size
    batch=16,                      # Batch size (adjust based on GPU memory)
    name='ball_segmentation',      # Experiment name
    patience=20,                   # Early stopping patience
    save=True,                     # Save checkpoints
    device=0,                      # Use GPU 0
    workers=2,                     # Number of dataloader workers
    project='runs/segment',        # Project directory
    exist_ok=True,                 # Overwrite existing project
    pretrained=True,               # Use pretrained weights
    optimizer='AdamW',             # Optimizer
    lr0=0.001,                     # Initial learning rate
    lrf=0.01,                      # Final learning rate factor
    momentum=0.937,                # Momentum
    weight_decay=0.0005,           # Weight decay
    warmup_epochs=3,               # Warmup epochs
    warmup_momentum=0.8,           # Warmup momentum
    box=7.5,                       # Box loss weight
    cls=0.5,                       # Classification loss weight
    dfl=1.5,                       # DFL loss weight
    plots=True,                    # Save training plots
    val=True,                      # Validate during training
)

print("\\n" + "="*60)
print("Training complete!")
print("="*60)
print(f"Best model saved to: runs/segment/ball_segmentation/weights/best.pt")
"""

# ============================================================================
# CELL 5: Evaluate Model
# ============================================================================
CELL_5_EVALUATE = """
from ultralytics import YOLO

# Load the best trained model
model = YOLO('runs/segment/ball_segmentation/weights/best.pt')

# Validate on validation set
metrics = model.val()

print("\\nValidation Metrics:")
print(f"mAP50: {metrics.seg.map50:.4f}")
print(f"mAP50-95: {metrics.seg.map:.4f}")
print(f"Precision: {metrics.seg.mp:.4f}")
print(f"Recall: {metrics.seg.mr:.4f}")
"""

# ============================================================================
# CELL 6: Test on Sample Images
# ============================================================================
CELL_6_TEST = """
from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow
import glob

# Load trained model
model = YOLO('runs/segment/ball_segmentation/weights/best.pt')

# Get some validation images
val_images = glob.glob('/content/ball_dataset/images/val/*.jpg')[:5]

print(f"Testing on {len(val_images)} validation images:\\n")

for img_path in val_images:
    # Run inference
    results = model(img_path, conf=0.25)

    # Visualize results
    annotated = results[0].plot()

    print(f"Image: {img_path}")
    print(f"Detections: {len(results[0].boxes)}")
    cv2_imshow(annotated)
    print("-" * 60)
"""

# ============================================================================
# CELL 7: Export Model
# ============================================================================
CELL_7_EXPORT = """
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/segment/ball_segmentation/weights/best.pt')

# Export to different formats
print("Exporting model to various formats...")

# Export to ONNX (for deployment)
model.export(format='onnx')
print("✓ Exported to ONNX")

# Export to TorchScript (for PyTorch deployment)
model.export(format='torchscript')
print("✓ Exported to TorchScript")

# Export to CoreML (for iOS/macOS)
# model.export(format='coreml')
# print("✓ Exported to CoreML")

print("\\nExported models saved in: runs/segment/ball_segmentation/weights/")
"""

# ============================================================================
# CELL 8: Download Trained Model
# ============================================================================
CELL_8_DOWNLOAD = """
from google.colab import files
import shutil

# Create a zip file with all model files
!zip -r ball_segmentation_model.zip runs/segment/ball_segmentation/weights/

# Download the zip file
files.download('ball_segmentation_model.zip')

print("Model downloaded! Extract and use best.pt for inference.")
"""

# ============================================================================
# CELL 9: Inference Example (for integration)
# ============================================================================
CELL_9_INFERENCE = """
from ultralytics import YOLO
import numpy as np

# Load model
model = YOLO('runs/segment/ball_segmentation/weights/best.pt')

def detect_ball_segmentation(frame, conf_threshold=0.25):
    \"\"\"
    Detect ball using YOLOv8 segmentation.

    Args:
        frame: Input frame (numpy array)
        conf_threshold: Confidence threshold

    Returns:
        tuple: (x, y, radius, mask) or None
    \"\"\"
    results = model(frame, conf=conf_threshold, verbose=False)

    if len(results) > 0 and len(results[0].boxes) > 0:
        # Get the detection with highest confidence
        boxes = results[0].boxes
        masks = results[0].masks

        if masks is not None:
            confidences = boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)

            # Get bounding box
            box = boxes.xyxy[best_idx].cpu().numpy()
            x1, y1, x2, y2 = box

            # Calculate center and radius
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            width = x2 - x1
            height = y2 - y1
            radius = int(np.sqrt(width**2 + height**2) / 2)

            # Get segmentation mask
            mask = masks.data[best_idx].cpu().numpy()

            return (cx, cy, radius, mask)

    return None

# Test the function
import cv2
test_img = cv2.imread('/content/ball_dataset/images/val/positive_0400.jpg')
result = detect_ball_segmentation(test_img)

if result:
    x, y, r, mask = result
    print(f"Ball detected at ({x}, {y}) with radius {r}")
    print(f"Mask shape: {mask.shape}")
else:
    print("No ball detected")
"""


def print_notebook_cells():
    """Print all cells for easy copy-paste into Colab."""

    print("="*80)
    print("GOOGLE COLAB YOLOV8 SEGMENTATION TRAINING NOTEBOOK")
    print("="*80)
    print("\nIMPORTANT: Enable GPU in Colab")
    print("Runtime > Change runtime type > Hardware accelerator > GPU")
    print("\nNote: YOLOv8 uses PyTorch (GPU), not TPU")
    print("="*80)

    cells = [
        ("Install Dependencies", CELL_1_INSTALL),
        ("Mount Google Drive (Optional)", CELL_2_MOUNT),
        ("Upload/Generate Dataset", CELL_3_DATASET),
        ("Train YOLOv8 Segmentation", CELL_4_TRAIN),
        ("Evaluate Model", CELL_5_EVALUATE),
        ("Test on Sample Images", CELL_6_TEST),
        ("Export Model", CELL_7_EXPORT),
        ("Download Trained Model", CELL_8_DOWNLOAD),
        ("Inference Example", CELL_9_INFERENCE),
    ]

    for i, (title, code) in enumerate(cells, 1):
        print(f"\n{'='*80}")
        print(f"CELL {i}: {title}")
        print(f"{'='*80}")
        print(code.strip())

    print(f"\n{'='*80}")
    print("END OF NOTEBOOK")
    print(f"{'='*80}")


if __name__ == "__main__":
    print_notebook_cells()
