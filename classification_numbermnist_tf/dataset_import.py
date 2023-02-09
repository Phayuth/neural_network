from tensorflow import keras
import numpy as np

(x_train_ori, y_train), (x_test_ori, y_test) = keras.datasets.mnist.load_data()

# Reshape dataset to desired value by mapping 0 to 255 to only 0 to 1 and into one dimensional image
x_train = x_train_ori.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test_ori.reshape(-1, 28 * 28).astype("float32") / 255.0