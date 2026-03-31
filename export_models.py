#!/usr/bin/env python3
"""
Export models to ONNX format.
Supports any YOLO model (YOLOv8, YOLOv9, YOLOv10, YOLO11, etc.) via ultralytics.

Usage:
    python export_models.py <model_name> [options]
    python export_models.py all
    
Examples:
    python export_models.py yolo26n
    python export_models.py yolov8n
    python export_models.py yolov9c
    python export_models.py yolov10s
    python export_models.py yolo11n
    python export_models.py yolo26n --size 1280
    python export_models.py yolo26s
    
Models will be exported to ./models/ directory.
"""

import os
import sys
import re
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Known non-YOLO models that need special handling
NON_YOLO_MODELS = {
    'rtdetr-l': {
        'name': 'RT-DETR Large',
        'filename': 'rtdetr-l.onnx',
        'url': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-l.onnx',
    },
    'rtdetr-x': {
        'name': 'RT-DETR XLarge',
        'filename': 'rtdetr-x.onnx',
        'url': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-x.onnx',
    },
}

# YOLO model patterns
YOLO_PATTERNS = [
    r'^yolo\d+n$',      # yolo26n, yolo11n
    r'^yolo\d+s$',      # yolo26s, yolo11s
    r'^yolo\d+m$',      # yolo26m, yolo11m
    r'^yolo\d+l$',      # yolo26l, yolo11l
    r'^yolo\d+x$',      # yolo26x, yolo11x
    r'^yolov\d+n$',     # yolov8n, yolov9n
    r'^yolov\d+s$',     # yolov8s, yolov9s
    r'^yolov\d+m$',     # yolov8m, yolov9m
    r'^yolov\d+l$',     # yolov8l, yolov9l
    r'^yolov\d+x$',     # yolov8x, yolov9x
    r'^yolov\d+c$',     # yolov9c
    r'^yolov\d+e$',     # yolov9e
]


def is_yolo_model(model_name: str) -> bool:
    """Check if the model name matches any YOLO pattern."""
    model_name = model_name.lower()
    for pattern in YOLO_PATTERNS:
        if re.match(pattern, model_name):
            return True
    return False


def get_model_type(model_name: str) -> str:
    """Get the type of model (yolo or other)."""
    if is_yolo_model(model_name):
        return 'yolo'
    if model_name in NON_YOLO_MODELS:
        return 'other'
    return 'unknown'


def export_yolo_model(model_name: str, output_dir: Path, imgsz: int = 640, half: bool = False) -> bool:
    """Export YOLO model to ONNX using ultralytics."""
    try:
        from ultralytics import YOLO
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle model name - add .pt if not present
        model_path = model_name
        if not model_path.endswith('.pt') and not model_path.endswith('.onnx'):
            model_path = f"{model_path}.pt"
        
        print(f"📥 Loading YOLO model: {model_path}")
        model = YOLO(model_path)
        
        # Determine output filename - append size suffix for non-default sizes
        base_name = model_name.replace('.pt', '')
        if imgsz != 640:
            # Non-default size: append to filename (e.g., yolo26s-1280.onnx)
            output_filename = f"{base_name}-{imgsz}.onnx"
        else:
            # Default size: no suffix (e.g., yolo26s.onnx)
            output_filename = f"{base_name}.onnx"
        output_path = output_dir / output_filename
        
        print(f"📦 Exporting to ONNX: {output_path}")
        print(f"   Input size: {imgsz}")
        
        # Export using ultralytics
        # The export method returns the path to the exported model
        # Use the output_dir directly as the project path
        exported_path = model.export(
            format='onnx',
            imgsz=imgsz,
            half=half,
            verbose=False,
            project=str(output_dir),
            name=model_name.replace('.pt', ''),
            exist_ok=True
        )
        
        # Check if export was successful - handle Path or string result
        if exported_path:
            exported_file = Path(exported_path)
            # If file was exported to current directory, move it to output_dir
            if exported_file.exists() and exported_file.parent != output_dir:
                target_path = output_dir / exported_file.name
                exported_file.rename(target_path)
                print(f"✅ Successfully exported: {model_name} -> {target_path}")
                return True
            elif exported_file.exists():
                # File is in the correct location (or already moved)
                print(f"✅ Successfully exported: {model_name} -> {output_path}")
                return True
        
        # Check if file exists at expected location
        if output_path.exists():
            print(f"✅ Successfully exported: {model_name}")
            return True
            
        # Check for any ONNX file in output directory related to this model
        for f in output_dir.glob(f"{model_name.replace('.pt', '')}*.onnx"):
            f.rename(output_path)
            print(f"✅ Successfully exported: {model_name}")
            return True
            
        print(f"❌ Export failed for {model_name}: file not found")
        return False
        
    except Exception as e:
        print(f"❌ Export failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_onnx_model(model_name: str, output_dir: Path) -> bool:
    """Download pre-exported ONNX model from ultralytics assets."""
    import urllib.request
    
    models_info = NON_YOLO_MODELS
    if model_name not in models_info:
        print(f"❌ No download URL for: {model_name}")
        return False
    
    info = models_info[model_name]
    output_path = output_dir / info['filename']
    
    if output_path.exists():
        print(f"✅ Model already exists: {output_path}")
        return True
    
    print(f"📥 Downloading {info['name']}...")
    print(f"   URL: {info['url']}")
    
    try:
        urllib.request.urlretrieve(info['url'], output_path)
        print(f"✅ Downloaded: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        # Clean up partial download
        if output_path.exists():
            output_path.unlink()
        return False


def export_all_models(output_dir: Path, imgsz: int = 640):
    """Export/download all supported models."""
    # First, try non-YOLO models
    for model_name in NON_YOLO_MODELS:
        print(f"\n{'='*50}")
        print(f"Processing: {model_name}")
        print('='*50)
        download_onnx_model(model_name, output_dir)
    
    print(f"\n{'='*50}")
    print(f"✅ All models processed")
    print(f"   Output directory: {output_dir}")
    print('='*50)


def list_supported_models():
    """List all supported model patterns."""
    print("Supported YOLO models (any variant/size):")
    print("  - yolo11n, yolo11s, yolo11m, yolo11l, yolo11x")
    print("  - yolov8n, yolov8s, yolov8m, yolov8l, yolov8x")
    print("  - yolov9n, yolov9s, yolov9c, yolov9m, yolov9l, yolov9e")
    print("  - yolov10n, yolov10s, yolov10m, yolov10l, yolov10x")
    print("  - yolo26n, yolo26s, yolo26m, yolo26l, yolo26x")
    print("\nOther supported models:")
    for name, info in NON_YOLO_MODELS.items():
        print(f"  {name:15} - {info['name']}")


def main():
    parser = argparse.ArgumentParser(
        description='Export YOLO models to ONNX format using ultralytics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_models.py yolo26n
  python export_models.py yolov8n
  python export_models.py yolov9c
  python export_models.py yolov10s
  python export_models.py yolo11n
  python export_models.py yolo26n --size 1280
  python export_models.py yolo26s
  python export_models.py rtdetr-l
  python export_models.py all
  python export_models.py --list
        """
    )
    parser.add_argument('model', nargs='?', help='Model name (e.g., yolo26n, yolov8s, rtdetr-l) or "all"')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output directory (default: ./models)')
    parser.add_argument('--size', '-s', type=int, default=640, help='Input image size (default: 640)')
    parser.add_argument('--half', action='store_true', help='Export with FP16 half precision')
    parser.add_argument('--list', '-l', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    # Default output directory is ./models subdirectory
    if args.output is None:
        output_dir = Path(__file__).parent / 'models'
    else:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.list:
        list_supported_models()
        return
    
    if args.model == 'all' or args.model is None:
        export_all_models(output_dir, args.size)
        return
    
    model_name = args.model.lower()
    model_type = get_model_type(model_name)
    
    print(f"\n{'='*50}")
    print(f"Exporting: {model_name}")
    print(f"   Type: {model_type}")
    print(f"   Size: {args.size}")
    print('='*50)
    
    if model_type == 'yolo':
        # Export YOLO model using ultralytics
        export_yolo_model(model_name, output_dir, imgsz=args.size, half=args.half)
    elif model_type == 'other':
        # Download pre-exported ONNX model
        download_onnx_model(model_name, output_dir)
    else:
        print(f"❌ Unknown model: {model_name}")
        print(f"   Use --list to see supported models")
        sys.exit(1)

if __name__ == '__main__':
    main()
