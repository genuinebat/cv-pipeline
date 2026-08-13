from cv_trainer import Trainer

if __name__ == "__main__":
    YAML = "./data.yaml"
    t = Trainer(yaml=YAML)
    t.train()
    t.save_onnx()