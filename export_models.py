#!/usr/bin/env python3
"""
Export models to ONNX or OpenVINO format.
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
    python export_models.py yolo26s --opset 12
    
Models will be exported to ./models/ directory.

OpenVINO format supports both FP16 (half) and INT8 (uint8) quantization.
ONNX format only supports FP16 (int8 is not supported for ONNX).
Use --opset to control the ONNX opset version when needed for downstream compatibility.
"""

import os
import sys
import re
import shutil
import argparse
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Known non-YOLO models that need special handling
NON_YOLO_MODELS = {
    'rtdetr-l': {
        'name': 'RT-DETR Large',
        'filename': 'rtdetr-l.onnx',
        'url': 'https://github.[AWS_SECRET_KEY_REDACTED]/v8.2.0/rtdetr-l.onnx',
    },
    'rtdetr-x': {
        'name': 'RT-DETR XLarge',
        'filename': 'rtdetr-x.onnx',
        'url': 'https://github.[AWS_SECRET_KEY_REDACTED]/v8.2.0/rtdetr-x.onnx',
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


def build_export_kwargs(
    *,
    export_format: str,
    imgsz: int,
    half: bool,
    int8: bool,
    end2end: bool,
    simplify: bool,
    output_dir: Path,
    model_name: str,
    opset: Optional[int],
) -> dict:
    """Build kwargs for ultralytics model.export()."""
    kwargs = {
        'format': export_format,
        'imgsz': imgsz,
        'half': half,
        'int8': int8,
        'end2end': end2end,
        'simplify': simplify,
        'verbose': False,
        'project': str(output_dir.resolve()),
        'name': model_name.replace('.pt', ''),
        'exist_ok': True,
    }
    if export_format == 'onnx' and opset is not None:
        kwargs['opset'] = opset
    return kwargs


def validate_export_options(export_format: str, opset: Optional[int]) -> None:
    """Validate export options before calling ultralytics."""
    if opset is None:
        return
    if export_format != 'onnx':
        raise ValueError('--opset is only supported for ONNX format')
    if opset <= 0:
        raise ValueError('--opset must be a positive integer')


def export_yolo_model(model_name: str, output_dir: Path, imgsz: int = 640,
                      half: bool = False, int8: bool = False,
                      export_format: str = 'onnx', end2end: bool = False,
                      simplify: bool = False, opset: Optional[int] = None) -> bool:
    """Export YOLO model to ONNX or OpenVINO using ultralytics."""
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
        
        # Determine output filename - append suffixes for non-default options
        base_name = model_name.replace('.pt', '')
        suffixes = []
        if simplify:
            suffixes.append('simplified')
        if imgsz != 640:
            suffixes.append(str(imgsz))
        
        # Determine precision suffix based on format and options
        if int8:
            # int8/uint8 is only supported for OpenVINO
            if export_format == 'onnx':
                print(f"⚠️  INT8 quantization is not supported for ONNX format.")
                print(f"   Use --format openvino to export with INT8 quantization.")
                return False
            suffixes.append('uint8')
        elif half:
            suffixes.append('half')
        
        # Build output filename based on format
        if export_format == 'openvino':
            # OpenVINO creates a directory with .xml and .bin files
            if suffixes:
                output_dirname = f"{base_name}-{'-'.join(suffixes)}"
            else:
                output_dirname = base_name
            output_path = output_dir / output_dirname
            output_filename = output_dirname  # for display
        else:
            # ONNX format
            if suffixes:
                output_filename = f"{base_name}-{'-'.join(suffixes)}.onnx"
            else:
                output_filename = f"{base_name}.onnx"
            output_path = output_dir / output_filename
        
        print(f"📦 Exporting to {export_format.upper()}: {output_path}")
        print(f"   Input size: {imgsz}")
        if int8:
            print(f"   Quantization: UINT8 (INT8)")
        elif half:
            print(f"   Precision: Half (FP16)")
        if export_format == 'onnx' and opset is not None:
            print(f"   ONNX opset: {opset}")
        
        # Export using ultralytics - use absolute path for project to ensure correct output location
        export_kwargs = build_export_kwargs(
            export_format=export_format,
            imgsz=imgsz,
            half=half,
            int8=int8,
            end2end=end2end,
            simplify=simplify,
            output_dir=output_dir,
            model_name=model_name,
            opset=opset,
        )
        try:
            exported_path = model.export(**export_kwargs)
        except TypeError as e:
            error_message = e.args[0] if e.args else str(e)
            if opset is not None and "unexpected keyword argument 'opset'" in error_message:
                print("⚠️  opset not supported by this ultralytics version; retrying without opset")
                export_kwargs.pop('opset', None)
                exported_path = model.export(**export_kwargs)
            else:
                raise
        
        if not exported_path:
            print(f"❌ Export failed for {model_name}: no output path returned")
            return False
        
        exported_path = Path(exported_path)
        
        if export_format == 'openvino':
            # OpenVINO exports to a directory
            if exported_path.exists() and exported_path.is_dir():
                # Move the exported directory to our target location
                if exported_path != output_path:
                    if output_path.exists():
                        shutil.rmtree(output_path)
                    shutil.move(str(exported_path), str(output_path))
                
                # Check for .xml and .bin files
                xml_file = output_path / f"{model_name}.xml"
                bin_file = output_path / f"{model_name}.bin"
                
                if xml_file.exists() and bin_file.exists():
                    print(f"✅ Successfully exported: {model_name}")
                    print(f"   📁 Model files: {output_dirname}/")
                    print(f"   📄 {xml_file.name}")
                    print(f"   📄 {bin_file.name}")
                    return True
                else:
                    # List what was actually created
                    print(f"⚠️  Expected .xml and .bin files not found in:")
                    for f in output_path.rglob('*'):
                        print(f"   {f}")
                    return True  # Consider it success if files exist
            else:
                print(f"❌ Export failed for {model_name}: OpenVINO directory not found")
                return False
        else:
            # ONNX format - use the path returned by model.export() directly
            if exported_path.exists() and exported_path.suffix == '.onnx':
                # Rename to our target if needed
                if exported_path != output_path:
                    if output_path.exists():
                        output_path.unlink()
                    exported_path.rename(output_path)
                
                print(f"✅ Successfully exported: {model_name} -> {output_path}")
                return True
            
            print(f"❌ Export failed for {model_name}: ONNX file not found at {exported_path}")
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


def export_all_models(output_dir: Path, imgsz: int = 640, export_format: str = 'onnx'):
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
    print("\nExport formats:")
    print("  - onnx: ONNX format (supports FP16/half, NOT int8)")
    print("  - openvino: Intel OpenVINO IR format (supports FP16/half AND uint8/int8)")


def main():
    parser = argparse.ArgumentParser(
        description='Export YOLO models to ONNX or OpenVINO format using ultralytics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ONNX format (FP16 supported, INT8 not supported)
  python export_models.py yolo26n
  python export_models.py yolov8n
  python export_models.py yolov9c
  python export_models.py yolov10s
  python export_models.py yolo11n
  python export_models.py yolo26n --size 1280
  python export_models.py yolo26s
  python export_models.py yolov8n --half
  python export_models.py rtdetr-l
  python export_models.py all
  python export_models.py --list

  # OpenVINO format (both FP16 and INT8 supported)
  python export_models.py yolo26n --format openvino
  python export_models.py yolo26n --format openvino --half
  python export_models.py yolo26n --format openvino --int8
  python export_models.py yolov8n --format openvino --size 640 --int8

  # Simplified ONNX model (adds -simplified suffix to output)
  python export_models.py yolo26n --simplify
  
  # Force a specific ONNX opset (useful for some OpenVINO/Frigate setups)
  python export_models.py yolo26s --opset 12
        """
    )
    parser.add_argument('model', nargs='?', help='Model name (e.g., yolo26n, yolov8s, rtdetr-l) or "all"')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output directory (default: ./models)')
    parser.add_argument('--size', '-s', type=int, default=640, help='Input image size (default: 640)')
    parser.add_argument('--half', action='store_true', help='Export with FP16 half precision')
    parser.add_argument('--int8', action='store_true', help='Export with INT8 quantization (only for OpenVINO)')
    parser.add_argument('--format', '-f', type=str, default='onnx', 
                        choices=['onnx', 'openvino'],
                        help='Export format (default: onnx)')
    parser.add_argument('--end2end', action='store_true', default=False, help='Export with end2end NMS')
    parser.add_argument('--opset', type=int, default=None,
                        help='ONNX opset version to export with (e.g., 12)')
    parser.add_argument('--simplify', action='store_true', default=False, 
                        help='Apply simplification to ONNX model (adds -simplified suffix to output)')
    parser.add_argument('--list', '-l', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    # Validate int8 with onnx format
    if args.int8 and args.format == 'onnx':
        print(f"❌ Error: INT8 quantization is not supported for ONNX format.")
        print(f"   Use --format openvino to export with INT8 quantization.")
        sys.exit(1)
    try:
        validate_export_options(args.format, args.opset)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
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
        export_all_models(output_dir, args.size, args.format)
        return
    
    model_name = args.model.lower()
    model_type = get_model_type(model_name)
    
    print(f"\n{'='*50}")
    print(f"Exporting: {model_name}")
    print(f"   Type: {model_type}")
    print(f"   Format: {args.format}")
    print(f"   Size: {args.size}")
    if args.opset is not None:
        print(f"   Opset: {args.opset}")
    print('='*50)
    
    if model_type == 'yolo':
        # Export YOLO model using ultralytics
        export_yolo_model(
            model_name, 
            output_dir, 
            imgsz=args.size, 
            half=args.half, 
            int8=args.int8,
            export_format=args.format,
            end2end=args.end2end,
            simplify=args.simplify,
            opset=args.opset,
        )
    elif model_type == 'other':
        # Download pre-exported ONNX model
        if args.format != 'onnx':
            print(f"⚠️  Only ONNX format is available for {model_name}, downloading ONNX")
        download_onnx_model(model_name, output_dir)
    else:
        print(f"❌ Unknown model: {model_name}")
        print(f"   Use --list to see supported models")
        sys.exit(1)

if __name__ == '__main__':
    main()
