# Vision Tests - Object Detection with ONNX Models

Object detection scripts using camera on Linux with RT-DETR, YOLOv9s, YOLOv10s (YOLO26s equivalent) models in ONNX format.

## Project Structure

```
vision-tests/
├── camera_detect.py    # Main script for camera-based object detection
├── export_models.py   # Script to export/download models to ONNX format
├── models/            # Directory for ONNX model files
├── pyproject.toml     # Project configuration
└── README.md          # This file
```

## Requirements

- Python 3.10+
- OpenCV (for camera access)
- ONNX Runtime (for inference)
- Ultralytics (for model export)

All dependencies are managed via `uv`:

```bash
# Install dependencies (already done in .venv)
uv pip install opencv-python onnxruntime ultralytics
```

## Setup

1. **Export/Download Models**

First, download or export the ONNX models:

```bash
# Download all available models
python export_models.py all

# Download specific model
python export_models.py yolov9s
python export_models.py rtdetr-l

# List available models
python export_models.py --list
```

Available models:
- `rtdetr-l` - RT-DETR Large
- `rtdetr-x` - RT-DETR XLarge  
- `yolov9s` - YOLOv9 Small
- `yolov9c` - YOLOv9 Medium
- `yolov10s` - YOLOv10 Small
- `yolov10n` - YOLOv10 Nano (equivalent to YOLO26s)

2. **List Available ONNX Models**

```bash
python camera_detect.py --list-models
```

## Usage

### Download/Export ONNX Models

Models are exported to the `./models/` directory:

```bash
# Export all available models
uv run export_models.py all

# Export specific model
uv run export_models.py yolov9s
uv run export_models.py rtdetr-l

# List available models
uv run export_models.py --list
```

Available models:
- `rtdetr-l` - RT-DETR Large (~125 MB)
- `rtdetr-x` - RT-DETR XLarge (~254 MB)
- `yolov9s` - YOLOv9 Small (~28 MB)
- `yolov9c` - YOLOv9 Medium (~97 MB)
- `yolov10s` - YOLOv10 Small (~28 MB)
- `yolov10n` - YOLOv10 Nano / YOLO26s equivalent (~9 MB)

### Camera Detection

Run object detection using your default camera:

```bash
uv run camera_detect.py --model yolov9s
```

Specify a different model:

```bash
uv run camera_detect.py --model rtdetr-l
uv run camera_detect.py --model yolov10n   # YOLO26s equivalent
```

Adjust detection parameters:

```bash
uv run camera_detect.py --model yolov9s --conf 0.5 --show-fps
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | Model name (without .onnx) | `yolov9s` |
| `--models-dir` | Directory containing ONNX models | `./models` |
| `--camera`, `-c` | Camera device index | Auto-detect |
| `--conf` | Confidence threshold (lower for more detections) | `0.25` |
| `--iou` | IoU threshold for NMS | `0.45` |
| `--show-fps` | Display FPS counter | Disabled |
| `--list-models` | List available ONNX models | Disabled |

**Note:** The default confidence threshold is 0.25. Some models (like YOLOv10n) output lower confidence scores, so you may need to lower this threshold (e.g., `--conf 0.1`) to get detections on simple test images. For real-world scenes with clear objects, 0.25 works well.

## Controls

- Press `q` or `Esc` to quit the detection window

## Environment

The scripts use the virtual environment created with `uv`:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run scripts
python camera_detect.py --model yolov9s
```

## Notes

- The camera detection requires a working camera device. Use `v4l2-ctl --list-devices` to check available cameras on Linux.
- ONNX Runtime will use CUDA if available, otherwise falls back to CPU.
- Models are loaded from the `./models/` directory by default.

## Example Usage

### Model export

```
uv run export_models.py yolo26s --size 320
uv run export_models.py yolo26m --size 320 --format openvino --int8
```

### Detection

```
uv run camera_detect.py --model yolo26m-320-uint8 --show-fps --image testdata/cam1_person.webp && open result.jpg
uv run camera_detect.py --model yolo26s-320 --show-fps --image testdata/cam1_person.webp && open result.jpg 
```

