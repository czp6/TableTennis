# YOLOv8 Segmentation Training Guide

## Quick Start

### Step 1: Generate Dataset Using Real Ball Photos

```bash
# Generate dataset from your video and labels
python real_ball_dataset_generator.py video.mp4 labels.json real_ball_dataset 500 100 100

# This will:
# 1. Extract real ball crops from your labeled video frames
# 2. Composite them onto various backgrounds
# 3. Apply transformations (scale, rotation, blur)
# 4. Generate 500 positive, 100 negative, 100 distractor samples
```

**Alternative: Synthetic balls (if you don't have labeled data)**
```bash
python synthetic_dataset_generator.py synthetic_ball_dataset 500 100 100
```

This creates:
```
synthetic_ball_dataset/
├── images/
│   ├── train/  (80% of data)
│   └── val/    (20% of data)
├── labels/
│   ├── train/  (segmentation polygons)
│   └── val/
├── data.yaml   (YOLO config)
└── dataset_info.json
```

### Step 2: Package for Google Colab

```bash
# Create a zip file to upload to Colab
cd synthetic_ball_dataset
zip -r ../ball_dataset.zip .
cd ..
```

### Step 3: Get Colab Training Code

```bash
# Print all Colab cells
python colab_train_yolo_segmentation.py > colab_cells.txt
```

### Step 4: Train in Google Colab

1. Open Google Colab: https://colab.research.google.com
2. Enable GPU: `Runtime > Change runtime type > GPU`
3. Copy-paste cells from `colab_cells.txt` or use the cells below

**Important**: YOLOv8 uses PyTorch (GPU), not TPU. TPUs are for TensorFlow/JAX.

## Colab Training Steps

### Cell 1: Install Dependencies
```python
!pip install ultralytics opencv-python-headless
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Cell 2: Upload Dataset
```python
from google.colab import files
uploaded = files.upload()  # Upload ball_dataset.zip
!unzip -q ball_dataset.zip -d /content/ball_dataset
```

### Cell 3: Train Model
```python
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # Nano model
results = model.train(
    data='/content/ball_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='ball_segmentation',
    patience=20,
    device=0
)
```

### Cell 4: Download Trained Model
```python
from google.colab import files
!zip -r model.zip runs/segment/ball_segmentation/weights/
files.download('model.zip')
```

## Dataset Details

### Positive Samples (Ball Only)
- Half-orange, half-white ball
- Random positions, sizes (15-60px radius)
- Random rotations (0-360°)
- Various backgrounds (table, floor, court, plain)

### Negative Samples (No Ball)
- Similar backgrounds
- Distractor objects (circles, ellipses, rectangles)
- Helps reduce false positives

### Distractor Samples (Ball + Distractors)
- Ball present with confusing objects
- Improves robustness

### Augmentations
- Gaussian blur
- Brightness/contrast variation
- Gaussian noise
- Random backgrounds

## Model Integration

After training, integrate the model into your tracker:

```python
from ultralytics import YOLO
import numpy as np

class YOLOSegmentationDetector:
    def __init__(self, model_path='best.pt'):
        self.model = YOLO(model_path)

    def detect(self, frame, conf_threshold=0.25):
        results = self.model(frame, conf=conf_threshold, verbose=False)

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            masks = results[0].masks

            if masks is not None:
                best_idx = np.argmax(boxes.conf.cpu().numpy())
                box = boxes.xyxy[best_idx].cpu().numpy()
                x1, y1, x2, y2 = box

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                radius = int(np.sqrt((x2-x1)**2 + (y2-y1)**2) / 2)

                mask = masks.data[best_idx].cpu().numpy()
                return (cx, cy, radius, mask)

        return None
```

## Tips

1. **More data = better results**: Generate 1000+ samples for production use
2. **Adjust confidence threshold**: Start with 0.25, tune based on results
3. **Monitor training**: Watch for overfitting (val loss increases)
4. **Early stopping**: Use patience=20 to stop if no improvement
5. **Export formats**: ONNX for deployment, TorchScript for PyTorch apps

## Troubleshooting

**Out of memory**: Reduce batch size (try 8 or 4)
**Slow training**: Use yolov8n-seg.pt (nano) instead of larger models
**Poor accuracy**: Generate more diverse training data
**False positives**: Add more negative samples

## Files Created

- `synthetic_dataset_generator.py` - Dataset generator
- `colab_train_yolo_segmentation.py` - Colab training script
- `YOLO_TRAINING_GUIDE.md` - This guide
