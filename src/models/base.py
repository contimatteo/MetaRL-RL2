from typing import Tuple

from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Dense, Input

###


def StandardPolicyNetwork(input_shape: Tuple[int], output_shape: Tuple[int]) -> Model:
    model = Sequential()

    model.add(Input(input_shape))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(output_shape, activation="softmax"))

    model.compile(optimizer="sgd", loss="categorical_crossentropy")

    return model


###


def StandardValueNetwork(input_shape: Tuple[int]) -> Model:
    model = Sequential()

    model.add(Input(input_shape))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="relu"))

    model.compile(optimizer="sgd", loss="mse")

    return model
