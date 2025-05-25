import tensorflow as tf

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model('/home/kaleab/PycharmProjects/UAV-Object-Detection-Distric4.0/tf_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the quantized model
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)