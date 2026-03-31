#!/usr/bin/env python3
"""
Object detection using camera with ONNX models.
Supports RT-DETR, YOLOv9s, YOLOv10s (YOLO26s).

Usage:
    python camera_detect.py --model yolov9s
    python camera_detect.py --model rtdetr-l --camera 0
    python camera_detect.py --model yolov10n --conf 0.5

Requirements:
    - ONNX model files in ./models/ directory
    - OpenCV for camera access
    - onnxruntime for inference
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

# Try to import required libraries
try:
    import cv2
    import onnxruntime as ort
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("   Install with: uv pip install opencv-python onnxruntime")
    sys.exit(1)

# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class ONNXDetector:
    """Object detector using ONNX models."""

    def __init__(self, model_path: str, conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45, input_size: Tuple[int, int] = (640, 640)):
        """
        Initialize the detector with an ONNX model.

        Args:
            model_path: Path to ONNX model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            input_size: Input image size (width, height)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        # Initialize ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        # Check if CUDA is available
        try:
            available = ort.get_available_devices()
            if 'CUDA' not in available:
                providers = ['CPUExecutionProvider']
        except:
            providers = ['CPUExecutionProvider']

        print(f"📦 Loading ONNX model: {model_path}")
        print(f"   Providers: {providers}")

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Get model info
        self.model_input_shape = self.session.get_inputs()[0].shape
        print(f"   Input shape: {self.model_input_shape}")
        print(f"   Output names: {self.output_names}")

        # Determine model type based on output structure
        self._detect_model_type()

    def _detect_model_type(self):
        """Detect model type from output structure."""
        # Get output shapes
        output_shapes = [self.session.get_outputs()[i].shape for i in range(len(self.output_names))]

        print(f"   Output shapes: {output_shapes}")

        # Detect based on output format:
        # YOLOv9: (1, 84, 8400) - flattened anchor format
        # YOLOv10: (1, 300, 6) - bbox + score format
        # RT-DETR: (1, 300, 84) - boxes + class scores

        if len(output_shapes) == 1:
            shape = output_shapes[0]
            if len(shape) == 3:
                _, num_boxes, num_classes = shape
                if num_classes == 6:
                    # YOLOv10 format: x1,y1,x2,y2,obj,class
                    self.model_type = 'yolov10'
                elif num_classes == 84:
                    # Could be RT-DETR (300, 84) or YOLO in different format
                    if num_boxes == 300:
                        self.model_type = 'rtdetr'
                    else:
                        self.model_type = 'yolo'
                else:
                    self.model_type = 'yolo'
            else:
                self.model_type = 'yolo'
        else:
            self.model_type = 'yolo'

        print(f"   Model type: {self.model_type}")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess image for inference using letterboxing."""
        # Get original and target shape
        original_h, original_w = image.shape[:2]
        target_h, target_w = self.input_size

        # Calculate scaling ratio
        ratio = min(target_w / original_w, target_h / original_h)

        # Calculate new dimensions and padding
        new_w = int(original_w * ratio)
        new_h = int(original_h * ratio)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        # Resize image with aspect ratio preservation
        if (new_w, new_h) != (original_w, original_h):
            resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized_img = image

        # Create a padded image (letterbox)
        padded_img = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded_img[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_img

        # Normalize and transpose
        rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)

        return batched, ratio, (pad_w, pad_h)

    def postprocess_yolo(self, outputs: List[np.ndarray],
                         ratio: float, pad: Tuple[int, int]) -> List:
        """Postprocess YOLO outputs.

        YOLO output format: [batch, num_classes+4, num_anchors] = (1, 84, 8400)
        - First 4 values: cx, cy, w, h (center format)
        - Value 4: objectness score (sigmoid applied)
        - Values 5-83: class scores (raw logits, need sigmoid)
        """
        # Output shape: (1, 84, 8400) -> transpose to (1, 8400, 84)
        predictions = outputs[0][0].T  # Now (8400, 84)

        # Extract raw values
        obj_scores_raw = predictions[:, 4:5]  # Objectness (8400, 1)
        class_scores_raw = predictions[:, 5:]  # Class scores (8400, 79)

        # Apply sigmoid to get proper probabilities
        obj_scores = 1.0 / (1.0 + np.exp(-obj_scores_raw))
        class_scores = 1.0 / (1.0 + np.exp(-class_scores_raw))
        max_class_conf = np.max(class_scores, axis=1, keepdims=True)  # (8400, 1)

        # Combined confidence score
        combined_scores = (obj_scores * max_class_conf).flatten()

        mask = combined_scores > self.conf_threshold
        filtered = predictions[mask]

        if len(filtered) == 0:
            return []

        # Extract boxes in center format (cx, cy, w, h)
        cx = filtered[:, 0]
        cy = filtered[:, 1]
        w = filtered[:, 2]
        h = filtered[:, 3]

        # Convert to corner format (x1, y1, x2, y2)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.column_stack([x1, y1, x2, y2])

        # Extract scores with sigmoid applied
        obj = filtered[:, 4]
        obj_sigmoid = 1.0 / (1.0 + np.exp(-obj))
        class_conf = np.max(1.0 / (1.0 + np.exp(-filtered[:, 5:])), axis=1)
        scores = obj_sigmoid * class_conf

        # Get class IDs from sigmoid-transformed scores
        class_scores_sigmoid = 1.0 / (1.0 + np.exp(-filtered[:, 5:]))
        class_ids = np.argmax(class_scores_sigmoid, axis=1)

        # Apply NMS
        indices = self._nms(boxes, scores, self.iou_threshold)

        pad_w, pad_h = pad
        
        results = []
        for idx in indices:
            box = boxes[idx]
            # Adjust for padding and scaling
            x1 = (box[0] - pad_w) / ratio
            y1 = (box[1] - pad_h) / ratio
            x2 = (box[2] - pad_w) / ratio
            y2 = (box[3] - pad_h) / ratio
            
            results.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(scores[idx]),
                'class_id': int(class_ids[idx]),
                'class_name': COCO_CLASSES[class_ids[idx]] if class_ids[idx] < len(COCO_CLASSES) else f'class_{class_ids[idx]}'
            })

        return results

    def postprocess_rtdetr(self, outputs: List[np.ndarray],
                           ratio: float, pad: Tuple[int, int]) -> List:
        """Postprocess RT-DETR outputs.

        RT-DETR output format: (1, num_boxes, 84) where:
        - First 4 values: cx, cy, w, h (normalized coordinates in 0-1 range)
        - Values 4-83: class scores (80 classes)
        """
        predictions = outputs[0][0]  # Shape: (300, 84), remove batch dimension

        # Extract boxes and scores
        center_x = predictions[:, 0]
        center_y = predictions[:, 1]
        width = predictions[:, 2]
        height = predictions[:, 3]
        class_scores = predictions[:, 4:]  # (300, 80)

        # Convert cx,cy,w,h to x1,y1,x2,y2
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        boxes = np.column_stack((x1, y1, x2, y2))

        # Get max scores and class IDs
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Filter by confidence
        mask = max_scores > self.conf_threshold
        boxes = boxes[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # Apply NMS
        indices = self._nms(boxes, max_scores, self.iou_threshold)

        pad_w, pad_h = pad

        results = []
        for idx in indices:
            box = boxes[idx]
            # Boxes are in normalized 0-1 range, scale to input image size first
            x1_scaled, y1_scaled, x2_scaled, y2_scaled = box
            x1_scaled *= self.input_size[0]
            y1_scaled *= self.input_size[1]
            x2_scaled *= self.input_size[0]
            y2_scaled *= self.input_size[1]

            # Adjust for padding and scaling
            x1_final = (x1_scaled - pad_w) / ratio
            y1_final = (y1_scaled - pad_h) / ratio
            x2_final = (x2_scaled - pad_w) / ratio
            y2_final = (y2_scaled - pad_h) / ratio

            results.append({
                'bbox': [int(x1_final), int(y1_final), int(x2_final), int(y2_final)],
                'confidence': float(max_scores[idx]),
                'class_id': int(class_ids[idx]),
                'class_name': COCO_CLASSES[class_ids[idx]] if class_ids[idx] < len(COCO_CLASSES) else f'class_{class_ids[idx]}'
            })

        return results

    def postprocess_yolov10(self, outputs: List[np.ndarray],
                           ratio: float, pad: Tuple[int, int]) -> List:
        """Postprocess YOLOv10 outputs.

        YOLOv10 output format: (1, 300, 6) where each row is:
        [x1, y1, x2, y2, obj_conf, class_id]
        """
        predictions = outputs[0][0]  # Remove batch dimension

        # Filter by confidence (obj_conf * class_conf)
        mask = predictions[:, 4] > self.conf_threshold
        filtered = predictions[mask]

        if len(filtered) == 0:
            return []

        # Extract boxes and scores
        boxes = filtered[:, :4]
        obj_scores = filtered[:, 4]
        class_ids = filtered[:, 5].astype(np.int32)

        # Combined score (already combined in YOLOv10)
        scores = obj_scores

        # Apply NMS
        indices = self._nms(boxes, scores, self.iou_threshold)

        pad_w, pad_h = pad

        results = []
        for idx in indices:
            box = boxes[idx]
            # Adjust for padding and scaling
            x1 = (box[0] - pad_w) / ratio
            y1 = (box[1] - pad_h) / ratio
            x2 = (box[2] - pad_w) / ratio
            y2 = (box[3] - pad_h) / ratio

            results.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(scores[idx]),
                'class_id': int(class_ids[idx]),
                'class_name': COCO_CLASSES[class_ids[idx]] if class_ids[idx] < len(COCO_CLASSES) else f'class_{class_ids[idx]}'
            })

        return results

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """Non-Maximum Suppression."""
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            intersection = w * h
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-6)

            mask = iou <= iou_threshold
            order = order[1:][mask]

        return np.array(keep, dtype=np.int32)

    def detect(self, image: np.ndarray) -> List:
        """Detect objects in image."""
        input_data, ratio, (pad_w, pad_h) = self.preprocess(image)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_data})

        # Postprocess based on model type
        if self.model_type == 'yolov10':
            return self.postprocess_yolov10(outputs, ratio, (pad_w, pad_h))
        elif self.model_type == 'yolo':
            return self.postprocess_yolo(outputs, ratio, (pad_w, pad_h))
        else:
            return self.postprocess_rtdetr(outputs, ratio, (pad_w, pad_h))


def draw_detections(image: np.ndarray, detections: List,
                    line_thickness: int = 2) -> np.ndarray:
    """Draw detection boxes on image."""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        label = f"{det['class_name']} {conf:.2f}"

        # Choose color based on class
        color = (0, 255, 0)  # Green for most objects

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

        # Draw label text
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image


def get_available_models(models_dir: str = './models') -> List[str]:
    """Get list of available ONNX models."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return []

    models = []
    for f in models_path.glob('*.onnx'):
        models.append(f.stem)

    return sorted(models)


def extract_size_from_model(model_name: str) -> Optional[Tuple[int, int]]:
    """Extract input size from model name using numeric suffix.

    Supports formats like:
    - model-640.onnx -> (640, 640)
    - model-640x480.onnx -> (640, 480)
    - model-000.onnx -> None (use default 640)
    - model.onnx (no suffix) -> None (use default 640)

    Args:
        model_name: Model name (with or without .onnx extension)

    Returns:
        Tuple of (width, height) or None if no size suffix found
    """
    import re

    # Remove .onnx extension if present
    name = model_name.replace('.onnx', '')

    # Match patterns like: name-640 or name-640x480
    # Note: 000 is treated as "no suffix" (use default)
    patterns = [
        r'-(\d{3,4})x(\d{3,4})$',  # name-640x480
        r'-(\d{3,4})$',             # name-640
    ]

    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            if len(match.groups()) == 2:
                # name-640x480 format
                width = int(match.group(1))
                height = int(match.group(2))
            else:
                # name-640 format (square)
                size = int(match.group(1))
                if size == 0:
                    # 000 means "use default" (not a valid size)
                    return None
                width = height = size

            return (width, height)

    return None


def find_camera(max_attempts: int = 10) -> Optional[int]:
    """Find available camera device."""
    for i in range(max_attempts):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            print(f"📷 Found camera at device {i}")
            return i
    return None


def get_parser():
    parser = argparse.ArgumentParser(
        description='Object detection using camera with ONNX models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python camera_detect.py --model yolov9s
  python camera_detect.py --model rtdetr-l --camera 0
  python camera_detect.py --model yolov10n --conf 0.5 --show-fps

Models are loaded from ./models/ directory.
Use export_models.py to download/export models first.
        '''
    )
    parser.add_argument('--model', '-m', type=str, default='yolov9s',
                        help='Model name (without .onnx extension)')
    parser.add_argument('--models-dir', type=str, default='./models',
                        help='Directory containing ONNX models')
    parser.add_argument('--camera', '-c', type=int, default=None,
                        help='Camera device index (default: auto-detect)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--size', '-s', type=str, default=None,
                        help='Input size (e.g., 640, 640x480). If not specified, '
                             'auto-detected from model name suffix (e.g., model-640.onnx). '
                             'Use 000 for default 640 (e.g., model-000.onnx)')
    parser.add_argument('--show-fps', action='store_true',
                        help='Display FPS counter')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models and exit')
    parser.add_argument('--image', type=str, help='Path to a single image file for detection')
    parser.add_argument('--output', type=str, default='result.jpg', help='Path to save the output image')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # List available models
    if args.list_models:
        models = get_available_models(args.models_dir)
        print("Available ONNX models:")
        if models:
            for m in models:
                print(f"  - {m}")
        else:
            print("  No models found in ./models/")
            print("  Run: python export_models.py all")
        return

    # Find model file
    model_path = Path(args.models_dir) / f"{args.model}.onnx"
    if not model_path.exists():
        # Try alternative paths
        alt_paths = [
            Path(args.models_dir) / f"{args.model}.onnx",
            Path.cwd() / args.models_dir / f"{args.model}.onnx",
            Path(__file__).parent / args.models_dir / f"{args.model}.onnx",
        ]

        for p in alt_paths:
            if p.exists():
                model_path = p
                break

        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            print(f"\nAvailable models in {args.models_dir}:")
            for m in get_available_models(args.models_dir):
                print(f"  - {m}")
            print(f"\nTo download models, run:")
            print(f"  python export_models.py all")
            sys.exit(1)

    # Determine input size
    input_size = None

    # First, try to get size from command line argument
    if args.size:
        # Parse size argument (e.g., "640" or "640x480")
        size_parts = args.size.lower().split('x')
        if len(size_parts) == 2:
            width = int(size_parts[0])
            height = int(size_parts[1])
            input_size = (width, height)
        else:
            size = int(size_parts[0])
            if size == 0:
                size = 640  # Default for 000
            input_size = (size, size)
        print(f"📐 Using CLI-specified input size: {input_size}")
    else:
        # Auto-detect from model name
        auto_size = extract_size_from_model(args.model)
        if auto_size:
            input_size = auto_size
            print(f"📐 Auto-detected input size from model name: {input_size}")
        else:
            input_size = (640, 640)  # Default
            print(f"📐 Using default input size: {input_size}")

    # Initialize detector
    detector = ONNXDetector(
        str(model_path),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=input_size
    )

    if args.image:
        # Process a single image
        image_path = Path(args.image)
        if not image_path.is_file():
            print(f"❌ Image file not found: {image_path}")
            sys.exit(1)

        print(f"🖼️  Processing image: {image_path}")
        frame = cv2.imread(str(image_path))

        # Run detection
        detections = detector.detect(frame)

        # Draw detections
        frame = draw_detections(frame, detections)

        # Save the output
        output_path = Path(args.output)
        cv2.imwrite(str(output_path), frame)
        print(f"✅ Saved result to: {output_path}")

    else:
        # Find camera
        camera_idx = args.camera if args.camera is not None else find_camera()
        if camera_idx is None:
            print("❌ No camera found. Use --image to process a file instead.")
            sys.exit(1)

        # Open camera
        print(f"📷 Opening camera {camera_idx}...")
        cap = cv2.VideoCapture(camera_idx)

        if not cap.isOpened():
            print(f"❌ Failed to open camera {camera_idx}")
            sys.exit(1)

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Get actual properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")

        # Detection loop
        print("\n🎯 Starting detection. Press 'q' to quit.")

        fps_counter = 0
        fps_start = time.time()
        current_fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break

            # Run detection
            detections = detector.detect(frame)

            # Draw detections
            frame = draw_detections(frame, detections)

            # Calculate FPS
            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()

            # Show FPS
            if args.show_fps:
                fps_text = f"FPS: {current_fps:.1f}"
                cv2.putText(frame, fps_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show detection count
            if detections:
                det_text = f"Objects: {len(detections)}"
                cv2.putText(frame, det_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display
            cv2.imshow('Object Detection', frame)

            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Done")


if __name__ == '__main__':
    main()
