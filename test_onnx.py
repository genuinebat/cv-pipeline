import cv2
import numpy as np
import onnxruntime as ort
import time

onnx_model_path = "trained/cv_model/weights/best.onnx"
confidence_threshold = 0.5
iou_threshold = 0.4

session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

input_details = session.get_inputs()[0]
input_name = input_details.name
input_shape = input_details.shape
input_h, input_w = input_shape[2], input_shape[3]

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
        
    original_h, original_w = frame.shape[:2]

    img_resized = cv2.resize(frame, (input_w, input_h))
    
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    img_chw = np.transpose(img_rgb, (2, 0, 1))
    
    input_tensor = np.expand_dims(img_chw, axis=0).astype(np.float32) / 255.0

    start_time = time.perf_counter()
        
    outputs = session.run(None, {input_name: input_tensor})
    
    end_time = time.perf_counter()
    inference_ms = (end_time - start_time) * 1000
    
    predictions = np.squeeze(outputs[0]).T
    
    boxes = []
    scores = []
    class_ids = []

    for row in predictions:
        classes_scores = row[4:]
        class_id = np.argmax(classes_scores)
        score = classes_scores[class_id]
        
        if score > confidence_threshold:
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            
            x_scale = original_w / input_w
            y_scale = original_h / input_h
            
            cx *= x_scale
            cy *= y_scale
            w *= x_scale
            h *= y_scale
            
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            
            boxes.append([x1, y1, int(w), int(h)])
            scores.append(float(score))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, iou_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            label = f"Class {class_ids[i]}: {scores[i]:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    fps_text = f"Inference: {inference_ms:.1f} ms"

    cv2.imshow("Pure ONNX Inference", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()