import os
import cv2
from ultralytics import YOLO

BEST_MODEL = "pothole_yolov8n3"

def predict_obj(img_path: str) -> str:

    os.makedirs("app/static/results", exist_ok=True)

    # Load model
    model = YOLO(f"ckpt/{BEST_MODEL}/weights/best.pt")

    # Predict
    filename = os.path.basename(img_path)
    result_path = os.path.join("app/static/results", filename)

    results = model.predict(
        source=img_path,
        conf=0.25
    )

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        probs = result.probs  # Probs object for classification outputs
        result.show()  # display to screen
        result.save(filename=result_path)

    return result_path