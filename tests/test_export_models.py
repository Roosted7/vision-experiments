import unittest
from pathlib import Path

import export_models


class TestExportModels(unittest.TestCase):
    def test_build_export_kwargs_includes_opset(self):
        kwargs = export_models.build_export_kwargs(
            export_format='onnx',
            imgsz=640,
            half=False,
            int8=False,
            end2end=False,
            simplify=False,
            output_dir=Path('models'),
            model_name='yolo26s',
            opset=12,
        )
        self.assertEqual(kwargs.get('opset'), 12)

    def test_build_export_kwargs_omits_opset_when_none(self):
        kwargs = export_models.build_export_kwargs(
            export_format='onnx',
            imgsz=640,
            half=False,
            int8=False,
            end2end=False,
            simplify=False,
            output_dir=Path('models'),
            model_name='yolo26s',
            opset=None,
        )
        self.assertNotIn('opset', kwargs)

    def test_build_export_kwargs_ignores_opset_for_openvino(self):
        kwargs = export_models.build_export_kwargs(
            export_format='openvino',
            imgsz=640,
            half=False,
            int8=False,
            end2end=False,
            simplify=False,
            output_dir=Path('models'),
            model_name='yolo26s',
            opset=12,
        )
        self.assertNotIn('opset', kwargs)

    def test_validate_export_options_accepts_onnx_opset(self):
        export_models.validate_export_options('onnx', 12)

    def test_validate_export_options_rejects_openvino_opset(self):
        with self.assertRaises(ValueError):
            export_models.validate_export_options('openvino', 12)

    def test_validate_export_options_rejects_non_positive_opset(self):
        with self.assertRaises(ValueError):
            export_models.validate_export_options('onnx', 0)
        with self.assertRaises(ValueError):
            export_models.validate_export_options('onnx', -1)

    def test_get_model_type(self):
        self.assertEqual(export_models.get_model_type('yolo26s'), 'yolo')
        self.assertEqual(export_models.get_model_type('rtdetr-l'), 'other')
        self.assertEqual(export_models.get_model_type('not-a-model'), 'unknown')


if __name__ == '__main__':
    unittest.main()
