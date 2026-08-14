import os
import yaml
from ultralytics import YOLO

class Trainer:
    def __init__(self, yaml):
        self._yaml = yaml
        self._model = YOLO('yolov8m.pt') 

    def train(self):
        results = self._model.train(
            data=self._yaml,
            epochs=3,
            imgsz=640,
            batch=4,
            device="cpu", # 0 = GPU, cpu = CPU
            project=os.path.join(os.getcwd(), "trained"),
            name='cv_model'
        )

    def save_onnx(self):
        best_model_path = os.path.join('trained', 'cv_model', 'weights', 'best.pt')
        best_model = YOLO(best_model_path)

        export_path = best_model.export(format='onnx', opset=12)