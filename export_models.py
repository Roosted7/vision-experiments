#!/usr/bin/env python3
"""
Export models to ONNX format.
Supports RT-DETR, YOLOv9s, YOLOv10s (YOLO26s).

Usage:
    python export_models.py [model_name]
    python export_models.py all
    
Models will be exported to ./models/ directory.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def get_model_info():
    """Return model configurations."""
    return {
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
        'yolov9s': {
            'name': 'YOLOv9 Small',
            'filename': 'yolov9s.onnx',
            'url': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9s.onnx',
        },
        'yolov9c': {
            'name': 'YOLOv9 Medium',
            'filename': 'yolov9c.onnx',
            'url': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c.onnx',
        },
        'yolov10s': {
            'name': 'YOLOv10 Small',
            'filename': 'yolov10s.onnx',
            'url': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10s.onnx',
        },
        'yolov10n': {
            'name': 'YOLOv10 Nano (YOLO26s equivalent)',
            'filename': 'yolov10n.onnx',
            'url': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.onnx',
        },
    }

def export_from_pytorch(model_name: str, output_dir: Path) -> bool:
    """Export model from PyTorch to ONNX using ultralytics."""
    try:
        from ultralytics import YOLO
        
        # Map model names to pt weights
        model_map = {
            'yolov9s': 'yolov9s.pt',
            'yolov9c': 'yolov9c.pt',
            'yolov10s': 'yolov10s.pt',
            'yolov10n': 'yolov10n.pt',
            'rtdetr-l': 'rtdetr-l.pt',
            'rtdetr-x': 'rtdetr-x.pt',
        }
        
        pt_file = model_map.get(model_name)
        if not pt_file:
            print(f"❌ No PyTorch mapping for: {model_name}")
            return False
            
        print(f"📥 Loading {model_name} from PyTorch...")
        model = YOLO(pt_file)
        
        # Export to ONNX in the output directory
        output_path = output_dir / f"{model_name}.onnx"
        print(f"📦 Exporting to ONNX: {output_path}")
        model.export(format='onnx', imgsz=640, verbose=False, project=str(output_dir), name=model_name)
        
        # The exported file is at output_dir/model_name.onnx
        if output_path.exists():
            print(f"✅ Successfully exported: {model_name}")
            # Clean up .pt file
            if pt_file.exists():
                pt_file.unlink()
            return True
        
        # Fallback: check if file was created elsewhere
        fallback_path = Path(f"{model_name}.onnx")
        if fallback_path.exists():
            fallback_path.rename(output_path)
            print(f"✅ Successfully exported (moved): {model_name}")
            return True
            
        print(f"❌ Export failed for {model_name}: file not found")
        return False
        
    except Exception as e:
        print(f"❌ Export failed for {model_name}: {e}")
        return False

def download_onnx_model(model_name: str, output_dir: Path) -> bool:
    """Download pre-exported ONNX model from ultralytics assets."""
    import urllib.request
    
    models_info = get_model_info()
    if model_name not in models_info:
        print(f"❌ Unknown model: {model_name}")
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

def export_all_models(output_dir: Path):
    """Export/download all supported models."""
    models = ['rtdetr-l', 'rtdetr-x', 'yolov9s', 'yolov9c', 'yolov10s', 'yolov10n']
    
    success_count = 0
    for model in models:
        print(f"\n{'='*50}")
        print(f"Processing: {model}")
        print('='*50)
        
        if download_onnx_model(model, output_dir):
            success_count += 1
        else:
            # Try PyTorch export as fallback
            print(f"⬇️ Trying PyTorch export...")
            if export_from_pytorch(model, output_dir):
                success_count += 1
    
    print(f"\n{'='*50}")
    print(f"✅ Exported {success_count}/{len(models)} models")
    print(f"   Output directory: {output_dir}")
    print('='*50)

def main():
    parser = argparse.ArgumentParser(description='Export models to ONNX format')
    parser.add_argument('model', nargs='?', help='Model name (e.g., yolov9s, rtdetr-l) or "all"')
    parser.add_argument('--output', '-o', type=str, default='./models', help='Output directory')
    parser.add_argument('--list', '-l', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.list:
        print("Available models:")
        for name, info in get_model_info().items():
            print(f"  {name:15} - {info['name']}")
        return
    
    if args.model == 'all' or args.model is None:
        export_all_models(output_dir)
    else:
        # Try direct download first
        if not download_onnx_model(args.model, output_dir):
            print(f"⬇️ Trying PyTorch export...")
            export_from_pytorch(args.model, output_dir)

if __name__ == '__main__':
    main()
