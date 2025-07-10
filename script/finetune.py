import argparse
import shutil
import os 
from ultralytics import YOLO

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Train YOLOv8 model on pothole dataset.")
    parser.add_argument('--model', type=str, default="yolov8n", help='Path to YOLO model file (e.g., yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    args = parser.parse_args()

    model_path_ckpt = f"ckpt/{args.model}.pt"
    model_path_local = f"./{args.model}.pt"
    if os.path.exists(model_path_ckpt):
        model = YOLO(model_path_ckpt)
    else:
        # Pull model (YOLO will download if not exists)
        model = YOLO(f"{args.model}.pt")

        # Move it to ckpt/
        if os.path.exists(model_path_local):
            shutil.move(model_path_local, model_path_ckpt)
    
    # Train model
    model.train(
        data='data/pothole_data/data.yaml', 
        epochs=args.epochs, 
        imgsz=args.imgsz, 
        project='ckpt', 
        name=f'pothole_{args.model}',
    )

    # Validate model
    metrics = model.val()
    print(f"Metrics: {metrics}")

if __name__ == "__main__":
    main()
