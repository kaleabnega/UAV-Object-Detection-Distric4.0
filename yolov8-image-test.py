from ultralytics import YOLO
import cv2  # Import OpenCV

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Run detection on an image
results = model("vehicle-and-person.jpg", save=True)

# Display the image using OpenCV
for result in results:
    img = result.plot()  # Get the annotated image
    cv2.imshow("YOLO Detection", img)  # Show the image
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Close the window properly



