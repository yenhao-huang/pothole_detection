import argparse
from ultralytics import YOLO
from torchinfo import summary

def main():
    parser = argparse.ArgumentParser(description="Run prediction and evaluation on pothole dataset using YOLO.")
    parser.add_argument('--model', type=str, default="yolo11x.pt", help='Model file name or path (e.g., yolov8n.pt or ckpt/best.pt)')
    args = parser.parse_args()

    # Load YOLO model
    model_path = f"ckpt/{args.model}" if not args.model.endswith(".pt") else args.model
    model = YOLO(model_path).model  # .model is the pytorch module

    # Display model summary
    summary(
        model,
        input_size=(16, 3, 640, 640),
        depth=3,
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        verbose=1
    )

if __name__ == "__main__":
    main()
