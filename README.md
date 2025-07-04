# YOLOv8 Pothole Detection

This project uses **Ultralytics YOLOv8** for pothole detection in images. It can be applied to road inspection, intelligent transportation systems, and related fields.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. repare data
Convert the dataset to COCO format and split into train / val:
```bash
python split_raw_data.py
```

### 3. Train the model
```bash
python script/finetune.py
```

### 4. Evaluate the model
```bash
python script/evaluate.py
```

---

## Data

* **Format conversion**: Converted from PASCAL VOC format to COCO format to match YOLOv8 requirements.
* **Dataset split**: Divided into train / val sets.
* **YOLOv8 requirements**: A `data.yaml` file is required to specify the dataset locations.

---

## Model Overview

| Attribute        | Value                                                |
| ---------------- | ---------------------------------------------------- |
| Parameters       | \~3.2 M                                        |
| Layers           | 33                                                   |
| Width multiplier | 0.25                                                 |
| Head             | Anchor-free, decoupled classification and regression |
| Architecture     | CSP bottlenecks, simplified head, no Focus layer     |


---

## Results

### Performance
| Metric                 | Value   |
| ---------------------- | ------- |
| Precision (B)          | 0.85399 |
| Recall (B)             | 0.70845 |
| mAP50 (B)              | 0.81022 |
| mAP50-95 (B)           | 0.55191 |

### Speed (ms per image)
| Stage       | Time (ms) |
| ----------- | --------- |
| Preprocess  | 0.60      |
| Inference   | 5.19      |
| Loss        | 0.00026   |
| Postprocess | 0.79      |

---

## 🔗 References

* [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
* [Kaggle Pothole Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/pothole-detection)