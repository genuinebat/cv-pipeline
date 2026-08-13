import os
import yaml
from ultralytics import YOLO

class Trainer:
    def __init__(self, _yaml):
        self._yaml = _yaml
        self._model = YOLO('yolov8m.pt') 

    def train(self):
        results = model.train(
            data=yaml,
            epochs=50,
            imgsz=640,
            batch=16,
            device=0, # 0 = GPU, cpu = CPU
            project='trained',
            name='cv_model'
        )

    def save_onnx(self):
        best_model_path = os.path.join('trained', 'cv_model', 'weights', 'best.pt')
        best_model = YOLO(best_model_path)

        export_path = best_model.export(format='onnx', opset=12)