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

    def test_get_model_type(self):
        self.assertEqual(export_models.get_model_type('yolo26s'), 'yolo')
        self.assertEqual(export_models.get_model_type('rtdetr-l'), 'other')
        self.assertEqual(export_models.get_model_type('not-a-model'), 'unknown')


if __name__ == '__main__':
    unittest.main()
