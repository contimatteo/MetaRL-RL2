from typing import Callable, Tuple, List

import tensorflow as tf

from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM

###


def AC_BackboneLayer() -> Callable[[tf.Tensor], Layer]:

    def backbone(input_x: tf.Tensor) -> tf.Tensor:
        x = Dense(512, activation="relu", kernel_initializer='he_uniform')(input_x)
        x = Dense(512, activation="relu", kernel_initializer='he_uniform')(x)
        return x

    return backbone


def AC_EncoderLayer() -> Callable[[tf.Tensor], Layer]:

    def encoder(input_x: tf.Tensor) -> tf.Tensor:
        # x = Flatten()(x)
        return input_x

    return encoder


def A_HeadLayer(n_actions: int, discrete: bool) -> Callable[[tf.Tensor], Layer]:
    activation = "softmax" if discrete else "linear"

    def head(input_x: tf.Tensor) -> tf.Tensor:
        x = Dense(n_actions, activation=activation, kernel_initializer='he_uniform')(input_x)
        return x

    return head


def C_HeadLayer() -> Callable[[tf.Tensor], Layer]:

    def head(input_x: tf.Tensor) -> tf.Tensor:
        x = Dense(1, activation='linear')(input_x)
        return x

    return head


def AC_MetaMemoryLayer(name: str) -> Callable[[tf.Tensor], Layer]:

    def memory(input_x: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        ### x -> (batch, trajectory_shape)
        x = tf.expand_dims(input_x, axis=1)  ### -> (batch, timestamps, trajectory_shape)
        ### -> (batch, 1, trajectory_shape)
        x, memory_state, carry_state = LSTM(64, return_state=True, stateful=True, name=name)(x)
        return x, [memory_state, carry_state]

    return memory
