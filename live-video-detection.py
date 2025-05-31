

# ---------------------------------------------------------------------------------------------------------
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter, load_delegate
from collections import Counter
import time

# ─── Config ────────────────────────────────────────────────────────────────
MODEL_PATH        = "./yolov8n_saved_model/yolov8n_int8.tflite"
LABELS_PATH       = "labels.txt"
CONFIDENCE_THRESH = 0.4
IOU_THRESHOLD     = 0.5
WEBCAM_INDEX      = 0
INPUT_SIZE        = 640   # square input

# ─── Load Class Labels ──────────────────────────────────────────────────────
with open(LABELS_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

# ─── Initialize Interpreter (XNNPACK if available) ─────────────────────────
try:
    xnn = load_delegate('libxnnpack_delegate.so')
    interpreter = Interpreter(model_path=MODEL_PATH,
                              experimental_delegates=[xnn])
    print("✅ XNNPACK delegate loaded")
except Exception:
    interpreter = Interpreter(model_path=MODEL_PATH)
    print("⚠️ XNNPACK delegate unavailable, using CPU")

interpreter.allocate_tensors()
inp_det  = interpreter.get_input_details()[0]
out_det  = interpreter.get_output_details()[0]

print("Input details :", inp_det)
print("Output details:", out_det)

# ─── IOU and NMS ────────────────────────────────────────────────────────────
def iou(b1, b2):
    x1,y1,x2,y2 = b1; X1,Y1,X2,Y2 = b2
    ix1,iy1 = max(x1,X1), max(y1,Y1)
    ix2,iy2 = min(x2,X2), min(y2,Y2)
    inter = max(ix2-ix1,0)*max(iy2-iy1,0)
    a1 = max(x2-x1,0)*max(y2-y1,0)
    a2 = max(X2-X1,0)*max(Y2-Y1,0)
    return inter/(a1+a2-inter+1e-6)

def nms(boxes, thr):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while boxes:
        b = boxes.pop(0); keep.append(b)
        boxes = [x for x in boxes if x[5]!=b[5] or iou(x[:4], b[:4])<thr]
    return keep

# ─── Main Loop ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

prev = time.time()
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]

    # Preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    inp = img.astype(np.float32)/255.0
    inp = np.expand_dims(inp,0)

    interpreter.set_tensor(inp_det['index'], inp)
    interpreter.invoke()

    # Get raw output of shape [1,84,8400]
    raw = interpreter.get_tensor(out_det['index'])[0]         # shape (84,8400)
    preds = raw.T                                          # shape (8400,84)

    boxes = []
    for det in preds:
        # det[0:4]=cx,cy,w,h; det[4:84]=80 class confidences
        class_confs = det[4:]
        cls_id = int(np.argmax(class_confs))
        conf   = class_confs[cls_id]
        if conf < CONFIDENCE_THRESH: continue

        cx, cy, bw, bh = det[0], det[1], det[2], det[3]
        x1 = (cx - bw/2) * w
        y1 = (cy - bh/2) * h
        x2 = (cx + bw/2) * w
        y2 = (cy + bh/2) * h

        boxes.append([max(0,x1), max(0,y1),
                      min(w,x2), min(h,y2),
                      conf, cls_id])

    # Apply NMS
    keep = nms(boxes, IOU_THRESHOLD)

    # Draw
    counts = Counter()
    for x1,y1,x2,y2,conf,cls in keep:
        label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
        counts[label]+=1
        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        cv2.putText(frame,f"{label} {conf:.2f}",(int(x1),int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    # Display counts
    y0=20
    for lbl,c in counts.items():
        cv2.putText(frame,f"{lbl}: {c}",(10,y0),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        y0+=25

    # FPS
    now = time.time()
    fps = 1/(now-prev); prev=now
    cv2.putText(frame,f"FPS: {fps:.1f}",(10,h-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

    cv2.imshow("YOLOv8 TFLite", frame)
    if cv2.waitKey(1)&0xFF==ord('q'): break

cap.release()
cv2.destroyAllWindows()





