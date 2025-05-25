try:
    import tensorflow.lite
    print("Using TensorFlow Lite's Interpreter")
except ImportError:
    try:
        import tflite_runtime.interpreter
        print("Using TFLite Runtime")
    except ImportError:
        print("Neither TensorFlow Lite nor TFLite Runtime found")
