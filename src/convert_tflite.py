"""Step 3: convert a trained Keras model (.h5) to TensorFlow Lite for the Raspberry Pi.

Usage:
    python src/convert_tflite.py models/stop_cnn.h5 models/stop_cnn.tflite
"""
import sys

from tensorflow import lite
from tensorflow.keras import models

if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    tflite_model = lite.TFLiteConverter.from_keras_model(models.load_model(src)).convert()
    with open(dst, 'wb') as f:
        f.write(tflite_model)
    print('Wrote', dst)
