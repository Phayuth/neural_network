from tensorflow import keras

# There are 3 Ways to create a model
# 1st Method
def Model(imgh,imgw):
    model = keras.Sequential(
        [
            keras.Input(shape=(imgh * imgw)),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(10, activation= "sigmoid"),
        ]
    )
    return model

# # 2nd Method
# Model = keras.Sequential()
# Model.add(keras.Input(shape=(784)))
# Model.add(keras.layers.Dense(512, activation="relu"))
# Model.add(keras.layers.Dense(256, activation="relu"))
# Model.add(keras.layers.Dense(10, activation = 'sigmoid'))

# # 3rd Method
# inputs = keras.Input(shape=(784))
# x = keras.layers.Dense(512, activation="relu")(inputs)
# x = keras.layers.Dense(256, activation="relu")(x)
# outputs = keras.layers.Dense(10, activation="softmax")(x)
# Model = keras.Model(inputs=inputs, outputs=outputs)