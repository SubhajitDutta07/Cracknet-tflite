from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2


cracknet_model = load_model('./cracknet_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(cracknet_model)
tflite_model = converter.convert()


with open('cracknet_model.tflite', 'wb') as f:
    f.write(tflite_model)
