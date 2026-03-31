#!/usr/bin/env python3
"""
Object detection using camera with ONNX or OpenVINO models.
Supports RT-DETR, YOLOv9s, YOLOv10s (YOLO26s), YOLO11, etc.

Usage:
    python camera_detect.py --model yolov9s
    python camera_detect.py --model rtdetr-l --camera 0
    python camera_detect.py --model yolov10n --conf 0.5
    python camera_detect.py --model yolo11n_openvino  # OpenVINO format

Requirements:
    - ONNX model files (.onnx) or OpenVINO IR files (.xml + .bin) in ./models/ directory
    - OpenCV for camera access
    - onnxruntime for ONNX inference
    - openvino for OpenVINO inference (optional)
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

# Try to import OpenVINO (optional)
try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

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
        - Value 4: objectness score 
        - Values 5-83: class scores

        Note: Some models (e.g., quantized/simplified) may have output values
        in a narrow range that appear to be sigmoid-applied or near-zero logits.
        """
        # Output shape: (1, 84, 8400) -> transpose to (1, 8400, 84)
        predictions = outputs[0][0].T  # Now (8400, 84)

        # Check value ranges to determine if sigmoid is already applied
        # or if the model output is in a narrow range (e.g., half-precision simplified)
        rest = predictions[:, 4:]  # objectness + class scores
        sigmoid_applied = rest.max() <= 1.0

        if sigmoid_applied:
            # For quantized/simplified models, scores may be in narrow range
            # Use column 4 as confidence directly and raw class logits for class ID
            scores = predictions[:, 4]
            class_scores_raw = rest[:, 1:]  # Skip index 4, use 5 onwards
            class_ids = np.argmax(class_scores_raw, axis=1)
        else:
            # Standard YOLO: apply sigmoid to raw logits
            obj_scores_raw = predictions[:, 4:5]  # Objectness (8400, 1)
            class_scores_raw = predictions[:, 5:]  # Class scores (8400, 79)

            # Apply sigmoid to get proper probabilities
            obj_scores = 1.0 / (1.0 + np.exp(-obj_scores_raw))
            class_scores = 1.0 / (1.0 + np.exp(-class_scores_raw))
            max_class_conf = np.max(class_scores, axis=1, keepdims=True)  # (8400, 1)

            # Combined confidence score
            scores = (obj_scores * max_class_conf).flatten()
            class_ids = np.argmax(class_scores, axis=1)

        mask = scores > self.conf_threshold
        filtered = predictions[mask]
        filtered_scores = scores[mask]

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

        # Get class IDs (need to recalculate if using sigmoid path)
        if sigmoid_applied:
            filtered_class_ids = class_ids[mask]
        else:
            class_scores_sigmoid = 1.0 / (1.0 + np.exp(-filtered[:, 5:]))
            filtered_class_ids = np.argmax(class_scores_sigmoid, axis=1)

        # Apply NMS
        indices = self._nms(boxes, filtered_scores, self.iou_threshold)

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
                'confidence': float(filtered_scores[idx]),
                'class_id': int(filtered_class_ids[idx]),
                'class_name': COCO_CLASSES[filtered_class_ids[idx]] if filtered_class_ids[idx] < len(COCO_CLASSES) else f'class_{filtered_class_ids[idx]}'
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


class OpenVINODetector:
    """Object detector using OpenVINO models."""

    def __init__(self, model_dir: str, conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45, input_size: Tuple[int, int] = (640, 640)):
        """
        Initialize the detector with an OpenVINO model.

        Args:
            model_dir: Path to OpenVINO model directory (contains .xml and .bin)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            input_size: Input image size (width, height)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        model_path = Path(model_dir)
        # Find any .xml file in the directory
        xml_files = list(model_path.glob('*.xml'))
        if not xml_files:
            raise FileNotFoundError(f"No .xml file found in {model_dir}")
        xml_file = xml_files[0]
        
        print(f"📦 Loading OpenVINO model: {xml_file}")
        
        # Initialize OpenVINO core
        core = ov.Core()
        
        # Read model
        self.model = core.read_model(str(xml_file))
        
        # Get input shape
        self.input_tensor = self.model.inputs[0]
        self.input_name = self.input_tensor.any_name
        input_shape = self.input_tensor.shape
        print(f"   Input shape: {input_shape}")
        
        # Compile model for CPU (can be changed to GPU)
        self.compiled_model = core.compile_model(self.model, "CPU")
        self.infer_request = self.compiled_model.create_infer_request()
        
        # Get output
        self.output_tensor = self.compiled_model.outputs[0]
        # Handle case where tensor has no name
        try:
            self.output_name = self.output_tensor.any_name
        except RuntimeError:
            self.output_name = None
        output_shape = self.output_tensor.shape
        print(f"   Output shape: {output_shape}")
        
        # Determine model type based on output structure
        self._detect_model_type(output_shape)

    def _detect_model_type(self, output_shape):
        """Detect model type from output shape."""
        print(f"   Output shapes: {[self.output_tensor.shape]}")
        
        # OpenVINO output format for YOLO:
        # (1, 84, 8400) - same as ONNX YOLO
        if len(output_shape) == 3:
            _, num_classes, num_anchors = output_shape
            # num_classes = 84 means 80 classes + 4 bbox coords
            if num_classes == 84:
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

        # Normalize and transpose - OpenVINO expects (1, 3, H, W)
        rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)

        return batched, ratio, (pad_w, pad_h)

    def postprocess_yolo(self, outputs: List[np.ndarray],
                         ratio: float, pad: Tuple[int, int]) -> List:
        """Postprocess YOLO outputs from OpenVINO.
        
        OpenVINO YOLO output format: [1, 84, num_anchors] or [84, num_anchors]
        """
        raw_output = outputs[0]
        
        # Handle both 2D and 3D output
        if raw_output.ndim == 3:
            raw_output = raw_output[0]  # Remove batch dimension
        
        predictions = raw_output.T  # (num_anchors, 84)

        # Check value ranges to determine if sigmoid is already applied
        first_4 = predictions[:, :4]
        rest = predictions[:, 4:]
        sigmoid_applied = rest.max() <= 1.0
        
        if sigmoid_applied:
            # For quantized models (int8), class scores are poorly discriminative
            # Use index 4 as confidence and pick class based on raw scores
            scores = predictions[:, 4]
            class_scores_raw = rest[:, 1:]  # Skip index 4, use 5 onwards
            
            # Use raw logits for class determination
            class_ids = np.argmax(class_scores_raw, axis=1)
        else:
            # Apply sigmoid to raw values
            obj_scores = 1.0 / (1.0 + np.exp(-predictions[:, 4]))
            class_scores = 1.0 / (1.0 + np.exp(-predictions[:, 5:]))
            max_class_conf = np.max(class_scores, axis=1)
            scores = obj_scores * max_class_conf
            class_ids = np.argmax(class_scores, axis=1)

        mask = scores > self.conf_threshold
        filtered_scores = scores[mask]
        filtered = predictions[mask]

        if len(filtered) == 0:
            return []

        # Extract boxes in center format (cx, cy, w, h)
        cx = filtered[:, 0]
        cy = filtered[:, 1]
        w = filtered[:, 2]
        h = filtered[:, 3]

        # Check if bbox is in pixel coordinates or normalized (0-1)
        if cx.max() > 1 or cy.max() > 1 or w.max() > 1 or h.max() > 1:
            # Already in pixel coordinates - no normalization needed
            pass
        else:
            # Normalized to 0-1, scale to input size
            cx = cx * self.input_size[0]
            cy = cy * self.input_size[1]
            w = w * self.input_size[0]
            h = h * self.input_size[1]

        # Convert to corner format (x1, y1, x2, y2)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.column_stack([x1, y1, x2, y2])

        # Filter class_ids to valid range
        filtered_class_ids = class_ids[mask]
        filtered_class_ids = np.clip(filtered_class_ids, 0, len(COCO_CLASSES) - 1)

        # Apply NMS
        indices = self._nms(boxes, filtered_scores, self.iou_threshold)

        pad_w, pad_h = pad
        
        results = []
        for idx in indices:
            box = boxes[idx]
            x1 = (box[0] - pad_w) / ratio
            y1 = (box[1] - pad_h) / ratio
            x2 = (box[2] - pad_w) / ratio
            y2 = (box[3] - pad_h) / ratio
            
            results.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(filtered_scores[idx]),
                'class_id': int(filtered_class_ids[idx]),
                'class_name': COCO_CLASSES[filtered_class_ids[idx]] if filtered_class_ids[idx] < len(COCO_CLASSES) else f'class_{filtered_class_ids[idx]}'
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
        outputs = self.infer_request.infer({self.input_name: input_data})
        output_data = list(outputs.values())

        # Postprocess
        return self.postprocess_yolo(output_data, ratio, (pad_w, pad_h))


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


def print_detections(detections: List, prefix: str = "  ") -> None:
    """Print detection results in a nice readable format."""
    if not detections:
        return
    
    print("\n" + "=" * 60)
    print("🔍 DETECTIONS")
    print("=" * 60)
    print(f"{'Class ID':<10} {'Label':<20} {'Confidence':<12}")
    print("-" * 60)
    
    for i, det in enumerate(detections, 1):
        class_id = det['class_id']
        label = det['class_name']
        conf = det['confidence']
        print(f"{class_id:<10} {label:<20} {conf:.2%}")
    
    print("-" * 60)
    print(f"Total objects detected: {len(detections)}")
    print("=" * 60)


def get_available_models(models_dir: str = './models') -> List[str]:
    """Get list of available ONNX and OpenVINO models."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return []

    models = []
    # ONNX models
    for f in models_path.glob('*.onnx'):
        models.append(f.stem)
    
    # OpenVINO models (directories with any .xml files)
    for d in models_path.iterdir():
        if d.is_dir():
            # Check for any .xml file in the directory
            xml_files = list(d.glob('*.xml'))
            if xml_files:
                models.append(d.name)

    return sorted(models)


def extract_size_from_model(model_name: str) -> Optional[Tuple[int, int]]:
    """Extract input size from model name using numeric suffix.

    Supports formats like:
    - model-640.onnx -> (640, 640)
    - model-640x480.onnx -> (640, 480)
    - model-000.onnx -> None (use default 640)
    - model.onnx (no suffix) -> None (use default 640)
    - model-half.onnx -> None (use default 640)
    - model-uint8.onnx -> None (use default 640)
    - model-simplified.onnx -> None (use default 640)
    - model-simplified-640.onnx -> (640, 640)

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
    # Also handles precision suffixes like -half, -uint8, -simplified before size
    patterns = [
        r'-(\d{3,4})x(\d{3,4})(?:-(?:half|uint8|simplified))*$',  # name-640x480 or name-640x480-half or name-640x480-simplified
        r'-(\d{3,4})(?:-(?:half|uint8|simplified))*$',             # name-640 or name-640-half or name-640-simplified
    ]

    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            num_groups = len(match.groups())
            if num_groups >= 2 and match.group(2) is not None:
                if num_groups == 4 and match.group(2):
                    # name-640x480 format with precision suffix
                    width = int(match.group(1))
                    height = int(match.group(2))
                elif num_groups == 2 or (num_groups >= 3 and match.group(2) and not match.group(3)):
                    # name-640 format (square)
                    size = int(match.group(1))
                    if size == 0:
                        # 000 means "use default" (not a valid size)
                        return None
                    width = height = size
                else:
                    continue
                return (width, height)
            elif num_groups == 1:
                # Single group: just the size (e.g., name-640)
                size = int(match.group(1))
                if size == 0:
                    # 000 means "use default" (not a valid size)
                    return None
                return (size, size)

    return None


def extract_precision_from_model(model_name: str) -> Optional[str]:
    """Extract precision type from model name.

    Supports formats like:
    - model-half.onnx -> 'half'
    - model-uint8.onnx -> 'uint8'
    - model-simplified.onnx -> None (simplified is not precision)
    - model-640-half.onnx -> 'half'
    - model-640-uint8.onnx -> 'uint8'
    - model-simplified-640-half.onnx -> 'half'
    - model.onnx -> None (default precision)

    Args:
        model_name: Model name (with or without .onnx extension)

    Returns:
        String indicating precision type ('half', 'uint8') or None for default
    """
    import re

    # Remove .onnx extension if present
    name = model_name.replace('.onnx', '')

    # Remove -simplified suffix first (it's not a precision type)
    name = re.sub(r'-simplified$', '', name)
    name = re.sub(r'-simplified-', '-', name)

    # Match patterns with precision suffix at the end
    patterns = [
        r'-((?:half|uint8))-(?:\d+)?$',  # half-320 or uint8-320
        r'-((?:half|uint8))$',        # half or uint8 at end
    ]

    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            if match.group(1):
                return match.group(1)
            # Check if -half or -uint8 is in the match
            if 'half' in match.group(0):
                return 'half'
            elif 'uint8' in match.group(0):
                return 'uint8'

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
        print("Available models (ONNX and OpenVINO):")
        if models:
            for m in models:
                print(f"  - {m}")
        else:
            print("  No models found in ./models/")
            print("  Run: python export_models.py all")
        return

    # Find model file - ONNX or OpenVINO format
    model_path = None
    model_format = None
    
    # First try ONNX format
    onnx_path = Path(args.models_dir) / f"{args.model}.onnx"
    if onnx_path.exists():
        model_path = onnx_path
        model_format = 'onnx'
    
    # Try with -simplified suffix if not found
    if model_path is None:
        simplified_path = Path(args.models_dir) / f"{args.model}-simplified.onnx"
        if simplified_path.exists():
            model_path = simplified_path
            model_format = 'onnx'
    
    # Try alternative paths for ONNX if not found
    if model_path is None:
        alt_paths = [
            Path(args.models_dir) / f"{args.model}.onnx",
            Path(args.models_dir) / f"{args.model}-simplified.onnx",
            Path.cwd() / args.models_dir / f"{args.model}.onnx",
            Path.cwd() / args.models_dir / f"{args.model}-simplified.onnx",
            Path(__file__).parent / args.models_dir / f"{args.model}.onnx",
            Path(__file__).parent / args.models_dir / f"{args.model}-simplified.onnx",
        ]
        
        for p in alt_paths:
            if p.exists():
                model_path = p
                model_format = 'onnx'
                break
    
    # Try OpenVINO format (directory with .xml and .bin)
    if model_path is None:
        openvino_path = Path(args.models_dir) / args.model
        if openvino_path.is_dir():
            # Find any .xml file in the directory
            xml_files = list(openvino_path.glob('*.xml'))
            if xml_files:
                model_path = openvino_path
                model_format = 'openvino'
    
    # Try alternative paths for OpenVINO
    if model_path is None:
        alt_openvino_paths = [
            Path.cwd() / args.models_dir / args.model,
            Path(__file__).parent / args.models_dir / args.model,
        ]
        for p in alt_openvino_paths:
            if p.is_dir():
                xml_files = list(p.glob('*.xml'))
                if xml_files:
                    model_path = p
                    model_format = 'openvino'
                    break

    if model_path is None:
        print(f"❌ Model not found: {args.model}.onnx or {args.model}/*.xml")
        print(f"\nAvailable models in {args.models_dir}:")
        for m in get_available_models(args.models_dir):
            print(f"  - {m}")
        print(f"\nTo download models, run:")
        print(f"  python export_models.py all")
        sys.exit(1)
    
    print(f"📦 Using {model_format.upper()} model: {model_path}")

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

    # Auto-detect precision type from model name
    precision = extract_precision_from_model(args.model)
    if precision:
        print(f"⚡ Detected precision type: {precision}")

    # Initialize detector based on format
    if model_format == 'onnx':
        detector = ONNXDetector(
            str(model_path),
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            input_size=input_size
        )
    elif model_format == 'openvino':
        if not OPENVINO_AVAILABLE:
            print("❌ OpenVINO is not installed.")
            print("   Install with: uv pip install openvino")
            sys.exit(1)
        detector = OpenVINODetector(
            str(model_path),
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            input_size=input_size
        )
    else:
        print(f"❌ Unknown model format: {model_format}")
        sys.exit(1)

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

        # Print detection results to console
        print_detections(detections)

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
        
        # Track previous detection state to avoid printing every frame
        prev_detect_count = 0
        prev_detect_labels = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break

            # Run detection
            detections = detector.detect(frame)
            
            # Print detections when they change (new objects detected or cleared)
            current_labels = set(d['class_name'] for d in detections)
            current_count = len(detections)
            if current_count != prev_detect_count or current_labels != prev_detect_labels:
                print_detections(detections)
                prev_detect_count = current_count
                prev_detect_labels = current_labels

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
