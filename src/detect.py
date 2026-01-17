from ultralytics import YOLO
import cv2
from utils import ensure_dir, draw_label

ensure_dir("../outputs/images")

model = YOLO("../models/yolov8n.pt")
image_path = "../data/images/sample.jpg"

img = cv2.imread(image_path)
results = model(img)

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = box.conf[0]
        label = model.names[cls]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        draw_label(img, f"{label} {conf:.2f}", x1, y1)

cv2.imwrite("../outputs/images/detected.jpg", img)
cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

