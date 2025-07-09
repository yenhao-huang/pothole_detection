# Pothole Detection

This project uses **Ultralytics YOLOv8** for pothole detection in images. It can be applied to road inspection, intelligent transportation systems, and related fields.

TODO
- [ ] Change Models: yolov8 v.s. yolov11
- [ ] Change Optimizer ADAM
- [ ] Summary Challenges
---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
Convert the dataset to COCO format and split into train / val:
```bash
python utils/split_raw_data.py
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

| Item              | Value                             |
| ----------------- | --------------------------------- |
| **#Parameters**   | \~3.2M                |
| **Optimizer**     | SGD      |
| **Learning Rate** | 0.01 |
| **Loss Function** | CIoU Loss + CLs Loss + DFL Loss        |
| **Batch Size**    | 16  |
| **Epochs**        | 100      |

### Loss Function
                                                            
| Component    | Description                                                                                                                                                           |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DFL Loss** | Distribution Focal Loss —  helps the model focus more on learning from positive samples|
| **Cls Loss** | Binary Cross-Entropy (BCE) loss — evaluates class prediction accuracy                              |
| **CIoU Loss** | IoU-based loss (e.g., CIoU or SIoU) — measures overlap quality between predicted and ground-truth boxes |


---

## Results

### Performance
| Metric        | YOLOv8  | YOLOv11 |
| ------------- | ------- | ------- |
| Precision (B) | 0.85399 | 0.81587 |
| Recall (B)    | 0.70845 | 0.73650 |
| mAP50 (B)     | 0.81022 | 0.82713 |
| mAP50-95 (B)  | 0.55191 | 0.55596 |

### Speed (ms per image)
| Stage       | Time (ms) |
| ----------- | --------- |
| Preprocess  | 0.60      |
| Inference   | 5.19      |
| Loss        | 0.00026   |
| Postprocess | 0.79      |

## Challenges

---

## 🔗 References

* [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
* [Kaggle Pothole Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/pothole-detection)