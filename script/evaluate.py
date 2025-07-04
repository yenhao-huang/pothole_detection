from ultralytics import YOLO

model = YOLO('ckpt/pothole_yolov8/weights/best.pt')

results = model.predict(
    source='data/pothole_data/images/val', 
    save=True, 
    conf=0.25, 
    project='results', 
    name='pothole_test'
    )

metrics = model.val()
print(f"Metrics: {metrics}")