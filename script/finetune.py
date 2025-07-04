from ultralytics import YOLO

model = YOLO('ckpt/yolov8n.pt')  

model.train(
    data='data/pothole_data/data.yaml', 
    epochs=100, 
    imgsz=640, 
    project='ckpt', 
    name='pothole_yolov8'
)

metrics = model.val()
print(f"Metrics: {metrics}")