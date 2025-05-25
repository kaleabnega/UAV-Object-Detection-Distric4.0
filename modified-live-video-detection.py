import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from collections import Counter

# ─── Config ────────────────────────────────────────────────────────────────
MODEL_PATH        = "yolov8n_saved_model/yolov8n_int8.tflite"
WEBCAM_INDEX      = 0
INPUT_SIZE        = 640       # must match what you exported
NUM_THREADS       = 4         # leverage multiple CPU cores
CONF_THRESH       = 0.25      # final confidence threshold
IOU_THRESH        = 0.45      # NMS IoU threshold

# 80 COCO class names
COCO_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard',
    'cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase',
    'scissors','teddy bear','hair drier','toothbrush'
]

# ─── Load TFLite Interpreter ────────────────────────────────────────────────
interpreter = tflite.Interpreter(
    model_path=MODEL_PATH,
    num_threads=NUM_THREADS
)
interpreter.allocate_tensors()
inp_details  = interpreter.get_input_details()
out_details  = interpreter.get_output_details()

# ─── Helpers ────────────────────────────────────────────────────────────────
def preprocess(frame):
    """Resize & normalize BGR→RGB, add batch dim."""
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return np.expand_dims(img.astype(np.float32), axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def postprocess(output, orig_shape):
    """
    Decode raw YOLOv8‐style head output:
      output: (1,84,8400)
    Returns lists of (x,y,w,h), confidences, class_ids
    """
    data = sigmoid(np.squeeze(output))  # (84,8400)
    # split
    xywh      = data[0:4, :].T          # (8400,4)
    obj_conf  = data[4, :]              # (8400,)
    cls_prob  = data[5:, :]             # (80,8400)
    # final confidences
    cls_id    = np.argmax(cls_prob, axis=0)
    cls_conf  = cls_prob[cls_id, np.arange(cls_prob.shape[1])]
    confs     = obj_conf * cls_conf

    # filter by threshold
    mask      = confs > CONF_THRESH
    xywh      = xywh[mask]
    confs     = confs[mask]
    cls_id    = cls_id[mask].astype(int)

    # convert to x,y,w,h in pixels
    h0, w0 = orig_shape[:2]
    boxes = []
    for cx, cy, w, h in xywh:
        x1 = int((cx - w/2) * w0)
        y1 = int((cy - h/2) * h0)
        boxes.append([x1, y1, int(w*w0), int(h*h0)])  # for NMS: x,y,w,h

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, confs.tolist(), CONF_THRESH, IOU_THRESH)
    final = indices.flatten() if len(indices) else []
    return ([boxes[i] for i in final],
            [confs[i] for i in final],
            [cls_id[i] for i in final])

# ─── Run Live Inference ─────────────────────────────────────────────────────
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    inp = preprocess(frame)
    interpreter.set_tensor(inp_details[0]['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(out_details[0]['index'])

    boxes, scores, class_ids = postprocess(out, frame.shape)

    # draw & count
    counts = Counter()
    for (x, y, w, h), conf, cid in zip(boxes, scores, class_ids):
        counts[COCO_NAMES[cid]] += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"{COCO_NAMES[cid]} {conf:.2f}",
                    (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # overlay counts
    y0 = 20
    for label, cnt in counts.items():
        cv2.putText(frame, f"{label}: {cnt}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y0 += 25

    cv2.imshow("TFLite INT8 Real-Time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
