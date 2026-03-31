# Vision Tests - Object Detection with ONNX and OpenVINO Models

```diff
- WARNING: Vibe-coded repo for experiments. I am unable to provide any support. Use at your own risk!
```


## Project Structure

```
vision-tests/
├── camera_detect.py   # Main script for object detection in a camera stream or an image file
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
# Install dependencies
uv sync
```

## Setup

1. **Export/Download Models**

First, download or export the ONNX models:

```bash
# Download specific model
uv run export_models.py yolov9s
uv run export_models.py rtdetr-l

# YOLO, with an optional quantization
uv run export_models.py yolo26s --size 320
uv run export_models.py yolo26s --size 320 --half --simplify
uv run export_models.py yolo26m --size 320 --format openvino --int8
```

2. **List Available Models**

```bash
uv run camera_detect.py --list-models
```

## Usage

### Download/Export ONNX Models

Models are exported to the `./models/` directory:

### Camera Detection

Run object detection using your default camera:

```bash
uv run camera_detect.py --model yolov9s
```

Specify a different model:

```bash
uv run camera_detect.py --model rtdetr-l
```

Adjust detection parameters:

```bash
uv run camera_detect.py --model yolov9s --conf 0.5 --show-fps
```

### File Detection

```bash
uv run camera_detect.py --model yolo26s-320 --image testdata/good_cam1_person.webp && open result.jpg
uv run camera_detect.py --model yolo26m-320-uint8 --image testdata/good_cam1_person.webp && open result.jpg

uv run camera_detect.py --model yolo26s-320 --image testdata/bad_cam2-no-person.webp && open result.jpg
```

## Controls

- Press `q` or `Esc` to quit the detection window



