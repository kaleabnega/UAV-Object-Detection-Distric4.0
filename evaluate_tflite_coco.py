# import tensorflow as tf
# import numpy as np
# import json
# import time
# import cv2
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
#
#
# # Load TFLite model
# def load_tflite_model(model_path):
#     interpreter = tf.lite.Interpreter(model_path=model_path)
#     interpreter.allocate_tensors()
#     return interpreter
#
#
# # Preprocess image
# def preprocess_image(image_path, input_shape):
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (input_shape[1], input_shape[2]))
#     image = image.astype(np.float32) / 255.0
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image
#
#
# # Run inference
# def run_inference(interpreter, image):
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#
#     interpreter.set_tensor(input_details[0]['index'], image)
#     interpreter.invoke()
#
#     outputs = [interpreter.get_tensor(output['index']) for output in output_details]
#     return outputs
#
#
# # Convert detections to COCO format
# def convert_to_coco_format(image_id, outputs, input_size):
#     # boxes, scores, classes = outputs  # Adjust based on your model's output format
#
#     output_data = outputs[0]  # Remove the batch dimension
#
#     boxes = output_data[0, :4, :].T  # Shape (8400, 4) → (x, y, w, h)
#     scores = output_data[0, 4, :]  # Shape (8400,) → Objectness scores
#     class_probs = output_data[0, 5:, :]  # Shape (num_classes, 8400)
#
#     # Get class predictions
#     classes = np.argmax(class_probs, axis=0)  # Shape (8400,)
#
#     # Apply confidence threshold
#     conf_threshold = 0.3
#     valid_indices = scores > conf_threshold
#
#     boxes = boxes[valid_indices]
#     scores = scores[valid_indices]
#     classes = classes[valid_indices]
#
#     detections = []
#     for i in range(len(scores)):
#         if scores[i] > 0.3:  # Confidence threshold
#             box = boxes[i] * input_size  # Scale back to original size
#             detections.append({
#                 "image_id": image_id,
#                 "category_id": int(classes[i]),
#                 "bbox": [box[1], box[0], box[3] - box[1], box[2] - box[0]],
#                 "score": float(scores[i])
#             })
#     return detections
#
#
# # Evaluate model on COCO dataset
# def evaluate_tflite_model(model_path, coco_images_dir, coco_annotations):
#     interpreter = load_tflite_model(model_path)
#     input_details = interpreter.get_input_details()
#     input_size = input_details[0]['shape']
#
#     coco = COCO(coco_annotations)
#     image_ids = coco.getImgIds()
#     detections = []
#
#     for img_id in image_ids[:50]:  # Evaluate on 50 images for speed (modify if needed)
#         img_info = coco.loadImgs(img_id)[0]
#         img_path = f"{coco_images_dir}/{img_info['file_name']}"
#
#         image = preprocess_image(img_path, input_size)
#         # outputs = run_inference(interpreter, image)
#         # print(type(outputs), len(outputs), outputs)
#         # print(outputs[0].shape)  # Check the shape
#
#         # outputs = run_inference(interpreter, image)
#         #
#         # # Verify TFLite output format
#         # output_data = outputs[0]  # Shape should be (1, 84, 8400)
#         # print("Output shape:", output_data.shape)
#         # print("Sample data:", output_data[0, :5, :5])  # Inspect a small patch
#
#         outputs = run_inference(interpreter, image)
#         output_data = outputs[0]  # (1, 84, 8400)
#
#         # Extract confidence scores (assuming 80 classes)
#         confidence_scores = output_data[:, 4, :]  # 5th row contains object confidence scores
#
#         print("Max confidence score:", confidence_scores.max())
#         print("Top 5 confidence scores:", np.sort(confidence_scores.flatten())[-5:])
#
#         detections.extend(convert_to_coco_format(img_id, outputs, input_size[1]))
#
#     # Save detections
#     with open("detections.json", "w") as f:
#         json.dump(detections, f)
#
#     # Run COCO evaluation
#     coco_dt = coco.loadRes("detections.json")
#     coco_eval = COCOeval(coco, coco_dt, "bbox")
#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     coco_eval.summarize()
#
#
# # Run evaluation
# evaluate_tflite_model(
#     model_path="yolov8n_saved_model/yolov8n_int8.tflite",  # Path to your quantized TFLite model
#     coco_images_dir="coco/val2017",  # Path to COCO validation images
#     coco_annotations="coco/annotations/instances_val2017.json"  # COCO annotations
# )


# ---------------------------------------------------------------------------------------------------------------------------------------
# from PIL import Image
#
# import numpy as np
# import json
# import tensorflow.lite as tflite
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
#
# # Load COCO category mapping
# def load_coco_category_mapping(coco_annotations):
#     """
#     Loads COCO category mapping from class index to COCO category ID.
#     """
#     coco_gt = COCO(coco_annotations)
#     category_mapping = {i: cat_id for i, cat_id in enumerate(sorted(coco_gt.cats.keys()))}
#     return category_mapping
#
# # Convert model predictions to COCO format
# def convert_to_coco_format(image_id, outputs, input_size, category_mapping):
#     """
#     Converts model outputs into COCO format, correctly mapping category IDs.
#     """
#     output_data = outputs[0]  # Remove batch dimension
#
#     boxes = output_data[0, :4, :].T  # Extract bounding boxes
#     scores = output_data[0, 4, :]  # Extract confidence scores
#     class_probs = output_data[0, 5:, :]  # Extract class probabilities
#
#     classes = np.argmax(class_probs, axis=0)  # Get predicted class indices
#     conf_threshold = 0.3  # Confidence threshold
#
#     valid_indices = scores > conf_threshold
#     boxes = boxes[valid_indices]
#     scores = scores[valid_indices]
#     classes = classes[valid_indices]
#
#     detections = []
#     for i in range(len(scores)):
#         box = boxes[i] * input_size  # Scale bounding box to original image size
#         category_id = category_mapping.get(int(classes[i]), -1)  # Correct category mapping
#
#         if category_id == -1:
#             continue  # Skip invalid categories
#
#         detections.append({
#             "image_id": image_id,
#             "category_id": category_id,
#             "bbox": [box[1], box[0], box[3] - box[1], box[2] - box[0]],  # COCO format (x, y, width, height)
#             "score": float(scores[i])
#         })
#
#     return detections
#
# # Load COCO dataset annotations
# coco_annotations = "coco/annotations/instances_val2017.json"  # Update this path to your actual annotations file
# coco_gt = COCO(coco_annotations)
# category_mapping = load_coco_category_mapping(coco_annotations)
#
# # Load TensorFlow Lite model
# interpreter = tflite.Interpreter(model_path="yolov8n_saved_model/yolov8n_int8.tflite")  # Update with actual model path
# interpreter.allocate_tensors()
#
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# input_size = input_details[0]['shape'][1:3]
#
# # Run inference and collect results
# results = []
# for img_id in coco_gt.getImgIds():
#     img_info = coco_gt.loadImgs(img_id)[0]
#     image_path = f"coco/val2017/{img_info['file_name']}"  # Update this path
#     # image = np.array(Image.open(image_path).resize(input_size)) / 255.0  # Normalize
#
#     # image = np.array(Image.open(image_path).resize(input_size)) / 255.0  # Normalize
#     # image = np.expand_dims(image, axis=0)  # Add batch dimension
#     #
#     # print(f"Image shape after expansion: {image.shape}")  # Debugging line
#     #
#     # interpreter.set_tensor(input_details[0]['index'], image.astype(np.float32))  # Pass input tensor
#
#     # image = np.array(Image.open(image_path).resize(input_size)) / 255.0  # Normalize
#     #
#     # # Ensure the image has 3 channels
#     # if len(image.shape) == 3:  # (height, width, channels)
#     #     image = np.expand_dims(image, axis=0)  # Add batch dimension (1, height, width, channels)
#     # else:
#     #     print(f"Warning: Image {image_path} does not have 3 channels! Shape: {image.shape}")
#     #
#     # print(f"Image shape after expansion: {image.shape}")
#     #
#     # # Set the tensor with the expected input shape
#     # interpreter.set_tensor(input_details[0]['index'], image.astype(np.float32))  # Pass input tensor
#
#     image = np.array(Image.open(image_path).resize(input_size)) / 255.0  # Normalize
#
#     # Ensure the image has 3 channels
#     if len(image.shape) == 3 and image.shape[2] == 3:  # (height, width, channels)
#         image = np.expand_dims(image, axis=0)  # Add batch dimension (1, height, width, channels)
#     elif len(image.shape) == 2:  # (height, width)
#         # Convert grayscale image to RGB by repeating the channel
#         image = np.expand_dims(image, axis=-1)  # (height, width, 1)
#         image = np.repeat(image, 3, axis=-1)  # (height, width, 3)
#         image = np.expand_dims(image, axis=0)  # Add batch dimension (1, height, width, 3)
#     else:
#         print(f"Warning: Image {image_path} has unexpected number of channels! Shape: {image.shape}")
#
#     print(f"Image shape after expansion: {image.shape}")
#
#     # Set the tensor with the expected input shape
#     interpreter.set_tensor(input_details[0]['index'], image.astype(np.float32))  # Pass input tensor
#
#     # interpreter.set_tensor(input_details[0]['index'], np.expand_dims(image, axis=0).astype(np.float32))
#     interpreter.invoke()
#     outputs = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
#
#     img_detections = convert_to_coco_format(img_id, outputs, input_size[1], category_mapping)
#     results.extend(img_detections)
#
# # Save detections as JSON
# with open("detections.json", "w") as f:
#     json.dump(results, f)
#
# # Load results into COCO format
# coco_dt = coco_gt.loadRes("detections.json")
#
# # Run COCO evaluation
# coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
# coco_eval.evaluate()
# coco_eval.accumulate()
# coco_eval.summarize()

# --------------------------------------------------------------------------------------------------------------------------
# import numpy as np
# import tensorflow as tf
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# import cv2
# import os
#
# # Set confidence threshold (e.g., 0.5 or 0.6)
# conf_threshold = 0.5
#
# # Load the COCO annotations
# annFile = 'coco/annotations/instances_val2017.json'  # Replace with your COCO annotations file
# cocoGt = COCO(annFile)
# imgIds = cocoGt.getImgIds()
#
# # Load the model and the image
# model = tf.lite.Interpreter(model_path="yolov8n_saved_model/yolov8n_int8.tflite")
# model.allocate_tensors()
#
#
# def get_model_output(image_path):
#     # Load image
#     image = cv2.imread(image_path)
#     img_height, img_width, _ = image.shape
#
#     # Preprocess the image (resize to 640x640, which is the expected input size)
#     input_size = (640, 640)  # Set to 640x640 instead of 416x416
#     image_resized = cv2.resize(image, input_size)
#     image_resized = np.expand_dims(image_resized, axis=0).astype(np.float32)
#     image_resized /= 255.0  # Normalize if needed based on model training
#
#     # Run inference
#     input_details = model.get_input_details()
#     output_details = model.get_output_details()
#
#     model.set_tensor(input_details[0]['index'], image_resized)
#     model.invoke()
#
#     output_data = model.get_tensor(output_details[0]['index'])
#
#     return output_data, img_height, img_width
#
#
# def convert_box_to_coco_format(box, img_width, img_height):
#     # Convert normalized coordinates to COCO format [x_min, y_min, width, height]
#     x_center, y_center, width, height = box
#     x_min = (x_center - width / 2) * img_width
#     y_min = (y_center - height / 2) * img_height
#     width = width * img_width
#     height = height * img_height
#     return [x_min, y_min, width, height]
#
#
# def process_predictions(output_data, img_height, img_width):
#     boxes = output_data[0, :, :, 0:4]  # Get bounding boxes
#     confidences = output_data[0, :, :, 4]  # Get confidence score
#     class_probs = output_data[0, :, :, 5:]  # Get class probabilities
#
#     detections = []
#     for i in range(boxes.shape[0]):
#         for j in range(boxes.shape[1]):
#             if confidences[i, j] > conf_threshold:  # Apply confidence threshold
#                 box = boxes[i, j]
#                 class_scores = class_probs[i, j]
#                 class_id = np.argmax(class_scores)  # Get class with highest probability
#
#                 # Convert to COCO format
#                 coco_bbox = convert_box_to_coco_format(box, img_width, img_height)
#
#                 # Prepare the detection dictionary
#                 detection = {
#                     "image_id": image_id,
#                     "category_id": class_id,
#                     "bbox": coco_bbox,
#                     "score": float(confidences[i, j])
#                 }
#                 detections.append(detection)
#
#     return detections
#
#
# def evaluate_model(detections):
#     cocoDt = cocoGt.loadRes(detections)
#     cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
#     cocoEval.evaluate()
#     cocoEval.accumulate()
#     cocoEval.summarize()
#
#
# # Main evaluation loop
# def run_evaluation():
#     all_detections = []
#
#     for img_id in imgIds:
#         image_info = cocoGt.loadImgs(img_id)[0]
#         image_path = os.path.join('coco/val2017', image_info['file_name'])  # Replace with your image folder path
#
#         # Get model predictions
#         output_data, img_height, img_width = get_model_output(image_path)
#
#         # Process predictions
#         detections = process_predictions(output_data, img_height, img_width)
#         all_detections.extend(detections)
#
#     # Run evaluation on all detections
#     evaluate_model(all_detections)
#
#
# if __name__ == '__main__':
#     run_evaluation()

# ---------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
import os

# Confidence & NMS thresholds
CONF_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.45

# Load COCO annotations and build class-ID map
annFile = 'coco/annotations/instances_val2017.json'
cocoGt = COCO(annFile)
imgIds = cocoGt.getImgIds()
coco_cat_ids = sorted(cocoGt.cats.keys())  # e.g. [1,2,3,...,90]
id_map = {i: coco_cat_ids[i] for i in range(len(coco_cat_ids))}

# Load TFLite model
model = tf.lite.Interpreter(model_path="yolov8n_saved_model/yolov8n_int8.tflite")
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()

# Quantization params
input_scale, input_zero_point = input_details[0]['quantization']


def get_model_output(image_path):
    # Load image
    image = cv2.imread(image_path)
    img_height, img_width, _ = image.shape

    # Preprocess the image
    input_size = (640, 640)
    image_resized = cv2.resize(image, input_size)

    image_resized = image_resized.astype(np.float32) / 255.0  # normalize
    image_resized = np.expand_dims(image_resized, axis=0)  # add batch dimension

    # Run inference
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    model.set_tensor(input_details[0]['index'], image_resized)
    model.invoke()

    output_data = model.get_tensor(output_details[0]['index'])

    return output_data, img_height, img_width


def process_predictions(raw, img_w, img_h):
    # raw: (84,8400) → [4 coords, 1 obj, 79 class scores]
    coords = raw[0:4, :].T  # (8400,4): [x_center,y_center,w,h]
    obj_conf = raw[4, :].T  # (8400,)
    cls_conf = raw[5:, :].T  # (8400,79)

    # 1) Confidence filter
    mask = obj_conf > CONF_THRESHOLD
    coords = coords[mask]
    scores = obj_conf[mask]
    classes = cls_conf[mask].argmax(axis=1)

    # 2) Convert to pixel corner format
    boxes = []
    for (xc, yc, w, h) in coords:
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])  # x,y,w,h

    # 3) NMS
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores.tolist(),
        score_threshold=CONF_THRESHOLD,
        nms_threshold=NMS_IOU_THRESHOLD
    )
    if len(indices) == 0:
        return []

    keep = [i[0] for i in indices]

    detections = []
    for idx in keep:
        detections.append((boxes[idx], float(scores[idx]), int(classes[idx])))

    return detections


def run_evaluation():
    all_dets = []
    for img_id in imgIds:
        info = cocoGt.loadImgs(img_id)[0]
        path = os.path.join('coco/val2017', info['file_name'])

        raw_out, w, h = get_model_output(path)
        dets = process_predictions(raw_out, w, h)

        # Build COCO-format dicts
        for box, score, cls in dets:
            all_dets.append({
                "image_id": img_id,
                "category_id": id_map[cls],
                "bbox": [box[0], box[1], box[2], box[3]],
                "score": score
            })

    # Evaluate via pycocotools
    cocoDt = cocoGt.loadRes(all_dets)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    run_evaluation()

