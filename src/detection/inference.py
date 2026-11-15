from pathlib import Path
import cv2

def load_yolo(weights):
    from ultralytics import YOLO
    return YOLO(str(weights))

def predict_image(model, image_path, conf=0.25, iou=0.5):
    results = model.predict(source=str(image_path), conf=conf, iou=iou, verbose=False)
    result = results[0]
    names = result.names
    detections = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for box, score, cls in zip(boxes, confs, classes):
            detections.append({
                "class_id": int(cls),
                "class_name": names[int(cls)],
                "confidence": float(score),
                "box": [float(v) for v in box],
            })
    annotated = result.plot()
    return detections, annotated
