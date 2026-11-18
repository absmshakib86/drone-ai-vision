import os
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO = "video.mp4"
OUT = "blue_shirt_result2.mp4"
SNAP_DIR = "snapshots"
STRIDE = 3
BLUE_LOW = (90, 40, 20)       # গাঢ় নীলও ধরবে
BLUE_HIGH = (135, 255, 255)
MIN_BLUE_RATIO = 0.10

os.makedirs(SNAP_DIR, exist_ok=True)
model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Cannot open {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps / STRIDE, (W, H))


def blue_ratio(frame, box):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    tx1 = max(0, x1 + int(0.2 * bw))
    tx2 = min(W, x2 - int(0.2 * bw))
    ty1 = max(0, y1 + int(0.15 * bh))
    ty2 = min(H, y1 + int(0.55 * bh))
    torso = frame[ty1:ty2, tx1:tx2]
    if torso.size == 0:
        return 0.0
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
    return float(mask.mean()) / 255.0


locked_id = None
frame_no = 0
processed = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1
    if frame_no % STRIDE != 0:
        continue
    processed += 1

    results = model.track(frame, persist=True, classes=[0], conf=0.4, verbose=False)
    boxes = results[0].boxes

    people = []   # (id, box, blue_ratio)
    if boxes is not None and len(boxes) > 0:
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        for b, tid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            people.append((tid, (x1, y1, x2, y2), blue_ratio(frame, (x1, y1, x2, y2))))

    target = None
    if locked_id is not None:
        for p in people:
            if p[0] == locked_id:
                target = p
    if target is None and people:
        best = max(people, key=lambda p: p[2])
        if best[2] >= MIN_BLUE_RATIO:
            target = best
            locked_id = best[0]

    for p in people:
        tid, (x1, y1, x2, y2), ratio = p
        is_target = target is not None and p is target
        color = (0, 255, 0) if is_target else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_target else 1)
        cv2.putText(frame, f"id={tid} blue={ratio:.2f}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if target is not None:
        x1, y1, x2, y2 = target[1]
        cx = (x1 + x2) // 2
        error = cx - W // 2
        print(f"frame {frame_no}: TARGET id={target[0]} center_x={cx} error={error} blue={target[2]:.2f}")
    else:
        best_seen = max([p[2] for p in people], default=0.0)
        print(f"frame {frame_no}: no target (best blue={best_seen:.2f})")

    writer.write(frame)
    if processed % 10 == 0:
        cv2.imwrite(f"{SNAP_DIR}/frame_{frame_no:04d}.png", frame)

cap.release()
writer.release()
print("done ->", OUT, "and images in", SNAP_DIR)