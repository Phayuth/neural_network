from tensorflow import keras
from dataset_import import x_test, y_test, x_test_ori
# import matplotlib.pyplot as plt
import numpy as np

# Load Model
model = model = keras.models.load_model('./weight/model.h5')

# Test
model.evaluate(x_test, y_test, batch_size=32, verbose=2)

# Predict
predictions = model.predict(x_test)

# image = x_test_ori[0]
# fig = plt.figure
# plt.imshow(image, cmap='gray')
# plt.show()

print(np.argmax(predictions[0]))
print(y_test[0])