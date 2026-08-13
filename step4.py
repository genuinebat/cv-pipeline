from cv_trainer import Trainer

if __name__ == "__main__":
    YAML = "./data.yaml"
    t = Trainer(YAML)
    t.train()
    t.save_onnx()