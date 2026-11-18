from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("bus.jpg")

for r in results:
    for box in r.boxes:
        name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        print(f"{name} confidence={conf:.2f} center_x={cx:.0f}")