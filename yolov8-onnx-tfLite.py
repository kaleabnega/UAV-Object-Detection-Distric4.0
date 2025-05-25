# from ultralytics import YOLO
# import onnx
# from onnx_tf.backend import prepare
# import tensorflow as tf
#
# # 1️ Load YOLOv8 model and export to ONNX
# model = YOLO("yolov8n.pt")
# # Export to ONNX with Opset 16
# model.export(format="onnx", opset=16)  # Use opset=16 for compatibility
# print("✅ YOLOv8 exported to ONNX (Opset 16)!")
#
# # Load the ONNX model (with Opset 16)
# onnx_model = onnx.load("yolov8n.onnx")
#
# # Convert ONNX to TensorFlow
# tf_rep = prepare(onnx_model)
#
# # Save the TensorFlow model
# tf_rep.export_graph("tf_model")
#
# import tensorflow as tf
#
# # Convert to TFLite
# converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")
# tflite_model = converter.convert()
#
# # Save the final TinyML model
# with open("yolov8n.tflite", "wb") as f:
#     f.write(tflite_model)
#
# print("Conversion complete! TinyML model saved as yolov8n.tflite")



# ----------------------------------------------------------------------------------------------



# from ultralytics import YOLO
# print("✅ YOLOv8 is installed successfully!")


# from ultralytics import YOLO
#
# # Download the YOLOv8 Nano model if not available
# YOLO("yolov8n.pt")
# print("✅ YOLOv8n model is ready!")


from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Make sure this file is in your project folder

# Export directly to TFLite format
model.export(format="tflite")

print("✅ YOLOv8 successfully exported to TFLite!")


