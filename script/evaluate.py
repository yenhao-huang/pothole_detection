import argparse
from ultralytics import YOLO

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Run prediction and evaluation on pothole dataset using YOLOv8.")
    parser.add_argument('--model', type=str, default="pothole_yolo8", help='Model file name (e.g., yolov8n.pt or path to best.pt)')
    args = parser.parse_args()

    # Load model
    model = YOLO(f"ckpt/{args.model}/weights/best.pt")

    # Predict
    results = model.predict(
        source='data/pothole_data/images/val',
        save=True,
        conf=0.25,
        project='results',
        name='pothole_test'
    )

    # Evaluate
    metrics = model.val()
    print(f"Metrics: {metrics}")

if __name__ == "__main__":
    main()
