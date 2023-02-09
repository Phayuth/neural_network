from dataset_import import x_train,y_train
from model import Model
from tensorflow import keras
# import matplotlib.pyplot as plt

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Create Model
model = Model(28,28)

# Create a optimization and compile model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

# Train
history = model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)
print("Done")

# Save
model.save('./weight/model.h5')

# Plot train result
# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train'], loc='upper left')
# plt.show()

# plt.plot(history.history['accuracy'])
# plt.title('model accuracy')
# plt.ylabel('laccuracy')
# plt.xlabel('epoch')
# plt.legend(['train'], loc='upper left')
# plt.show()