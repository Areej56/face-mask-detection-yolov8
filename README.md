# Face Mask Detection using YOLOv8 😷

## Overview
This project implements a real-time **Face Mask Detection System** using **YOLOv8**.
The model detects and classifies faces into the following categories:
- Without Mask
- With Mask
- Mask Worn Incorrectly

The system is trained on a custom dataset converted from **Pascal VOC (XML)** format to **YOLO format** and supports inference on images, videos, and real-world data.

---

## Project Highlights
- Custom dataset preprocessing and annotation handling
- XML (Pascal VOC) → YOLO format conversion
- YOLOv8-based multi-class object detection
- Train / Validation / Test dataset splitting
- Real-world image inference
- Clean and modular project structure

---

## Tech Stack
- **Python**
- **YOLOv8 (Ultralytics)**
- **PyTorch**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **Pillow**

---

## Dataset
The dataset contains face images annotated with bounding boxes and labels:
- `without_mask`
- `with_mask`
- `mask_weared_incorrect`

Annotations were originally in XML format and converted to YOLO format for training.

> ⚠️ Note: Full dataset is not included in this repository.  
> Only sample images are provided for demonstration.

---

## Project Structure
face-mask-detection-yolov8/
│
├── data/
│ └── data.yaml
│
├── dataset/
│ └── sample_images/
│
├── src/
│ ├── xml_to_yolo.py
│ ├── split_dataset.py
│ ├── train.py
│ └── predict.py
│
├── results/
│ └── results.png
│
├── requirements.txt
├── README.md
└── .gitignore
---

## Installation
```bash
pip install -r requirements.txt

Training the Model
python src/train.py

The trained weights will be saved automatically in:

runs/detect/train/weights/best.pt
Inference

Run prediction on sample images:

python src/predict.py
Results

Example detection results:

##Author

##Areej Arslan
Machine Learning & Computer Vision Engineer (Entry-Level)

##License

This project is for educational and research purposes.
