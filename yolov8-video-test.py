from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open video file or webcam (use 0 for webcam)
cap = cv2.VideoCapture("pedestrian-video.mp4")  # Replace with "0" for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Run YOLO on each frame
    annotated_frame = results[0].plot()  # Get the annotated frame

    cv2.imshow("YOLO Video", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
