from typing import Callable

import tensorflow as tf

from tensorflow.python.keras.layers import Layer, Dense, Flatten

###


def AC_BackboneLayer() -> Callable[[tf.Tensor], Layer]:

    def backbone(x: tf.Tensor) -> Layer:
        x = Dense(512, activation="relu", kernel_initializer='he_uniform')(x)
        x = Dense(512, activation="relu", kernel_initializer='he_uniform')(x)
        return x

    return backbone


def AC_EncoderLayer() -> Callable[[tf.Tensor], Layer]:

    def encoder(x: tf.Tensor) -> Layer:
        x = Flatten()(x)
        return x

    return encoder


def A_HeadLayer(n_actions: int, discrete: bool) -> Callable[[tf.Tensor], Layer]:
    activation = "softmax" if discrete else "linear"

    def head(x: tf.Tensor) -> Layer:
        x = Dense(n_actions, activation=activation, kernel_initializer='he_uniform')(x)
        return x

    return head


def C_HeadLayer() -> Callable[[tf.Tensor], Layer]:

    def head(x: tf.Tensor) -> Layer:
        x = Dense(1, activation='linear')(x)
        return x

    return head
