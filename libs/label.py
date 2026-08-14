import os
import shutil
import cv2
import ollama
import json
import random
import math
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

class AutoLabeler:
    def __init__(self, ont, input_dir, validated_dir, labeled_dir, unlabeled_dir):
        _ont = CaptionOntology(ont)
        self._model = GroundingDINO(ontology=_ont)
        self._classes = list(_ont.classes()) 
        self._input_dir = input_dir
        self._validated_dir = validated_dir
        self._labeled_dir = labeled_dir
        self._unlabeled_dir = unlabeled_dir

    def label_and_validate(self):
        print("LABELING IN PROGRESS...please wait a moment")
        for filename in os.listdir(self._input_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(self._input_dir, filename)
            image = cv2.imread(img_path)
            if image is None: 
                continue
    
            img_h, img_w, _ = image.shape
    
            detections = self._model.predict(img_path)
    
            if len(detections.xyxy) < 1:
                print(f"{os.path.basename(img_path)} could not be labeled")
                shutil.move(img_path, os.path.join(self._unlabeled_dir, os.path.basename(img_path)))
                continue
    
            label_data = {
                "version": "0.3.3",
                "flags": {},
                "shapes": [],
                "imagePath": filename,
                "imageData": None,
                "imageHeight": img_h,
                "imageWidth": img_w
            }

            for i, box in enumerate(detections.xyxy):
                x_min, y_min, x_max, y_max = map(int, box)
                class_id = detections.class_id[i]
                predicted_class = self._classes[class_id]
                
                label_data["shapes"].append({
                    "label": predicted_class,
                    "text": "",
                    "points": [
                        [float(x_min), float(y_min)],
                        [float(x_max), float(y_max)]
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                })

            labeled_img_path = os.path.join(self._labeled_dir, filename)
            shutil.move(img_path, labeled_img_path)
    
            json_filename = filename.rsplit(".", 1)[0] + ".json"
            labeled_json_path = os.path.join(self._labeled_dir, json_filename)
    
            with open(labeled_json_path, "w", encoding="utf-8") as f:
                json.dump(label_data, f, indent=4)
            
            print(f"{filename} labeled successfully")

        
            is_valid_image = True
            
            for shape in label_data["shapes"]:
                pts = shape["points"]
                x_min, y_min = int(pts[0][0]), int(pts[0][1])
                x_max, y_max = int(pts[1][0]), int(pts[1][1])
                predicted_class = shape["label"]
                
                cropped_img = image[y_min:y_max, x_min:x_max]
                
                if cropped_img.size == 0:
                    is_valid_image = False
                    break
                    
                _, buffer = cv2.imencode('.png', cropped_img)
                img_bytes = buffer.tobytes()
                
                prompt = f"Is this image specifically a {predicted_class}? Answer exactly 'YES' or 'NO'."
                
                response = ollama.chat(
                    model='llava',
                    messages=[{'role': 'user', 'content': prompt, 'images': [img_bytes]}]
                )
                
                answer = response['message']['content'].strip().upper()
                
                if "NO" in answer or "YES" not in answer:
                    is_valid_image = False
                    break
                    
            if is_valid_image:
                print(f"{filename} passed secondary validation")
                validated_img_path = os.path.join(self._validated_dir, filename)
                validated_json_path = os.path.join(self._validated_dir, json_filename)
                
                shutil.move(labeled_img_path, validated_img_path)
                shutil.move(labeled_json_path, validated_json_path)
            else:
                print(f"{filename} failed secondary validation")

class LabelPreperation:
    def __init__(self, class_map):
        self._class_map = class_map

    def convert_validated_json_labels_to_text(self, input_dir, output_dir):
        for f in os.listdir(input_dir):
            if f.endswith(".jpg"):
                shutil.copy(os.path.join(input_dir, f), output_dir)
            elif f.endswith(".json"):
                json_f = os.path.join(input_dir, f)
                txt_f = os.path.join(output_dir, f.replace(".json", ".txt"))

                with open(json_f, "r") as jf:
                    data = json.load(jf)
                
                img_w, img_h = data["imageWidth"], data["imageHeight"]

                with open(txt_f, 'w') as file:
                    for shape in data['shapes']:
                        label = shape['label'].lower()
                        if label not in self._class_map:
                            continue
                            
                        class_id = self._class_map[label]
                        
                        points = shape['points']
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        
                        w = (x2 - x1) / img_w
                        h = (y2 - y1) / img_h
                        x_center = (x1 / img_w) + (w / 2)
                        y_center = (y1 / img_h) + (h / 2)
                        
                        file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
            else:
                continue

    def split_data(self, input_dir, output_train, output_val, output_test=None)-> tuple[int, int, int]:
        all_files = os.listdir(input_dir)
        image_files = [f for f in all_files if f.lower().endswith(".jpg")]
        
        paired_data = []
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            lbl_file = base_name + ".txt" 
            
            if lbl_file in all_files:
                paired_data.append((img_file, lbl_file))
                
        random.shuffle(paired_data)
        
        total_pairs = len(paired_data)
        train_count = math.floor(total_pairs * 0.8) 
        
        train_pairs = paired_data[:train_count]
        val_pairs = paired_data[train_count:]
        
        for img, lbl in train_pairs:
            shutil.copy2(os.path.join(input_dir, img), os.path.join(output_train, img))
            shutil.copy2(os.path.join(input_dir, lbl), os.path.join(output_train, lbl))
                
        for img, lbl in val_pairs:
            shutil.copy2(os.path.join(input_dir, img), os.path.join(output_val, img))
            shutil.copy2(os.path.join(input_dir, lbl), os.path.join(output_val, lbl))

        return len(train_pairs), len(val_pairs), 0