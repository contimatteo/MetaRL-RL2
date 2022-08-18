from typing import Callable, Tuple, List

import tensorflow as tf

from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Concatenate

###


def AC_BackboneLayer() -> Callable[[tf.Tensor], Layer]:

    def backbone(input_x: tf.Tensor) -> tf.Tensor:
        x = Dense(128, activation="relu", kernel_initializer='he_uniform')(input_x)
        # x = Dropout(.2)(x)
        x = Dense(128, activation="tanh", kernel_initializer='he_uniform')(x)
        # x = Dropout(.2)(x)
        x = Dense(128, activation="relu", kernel_initializer='he_uniform')(x)
        return x

    return backbone


def AC_EncoderLayer() -> Callable[[tf.Tensor], Layer]:

    def encoder(input_x: tf.Tensor) -> tf.Tensor:
        # x = Flatten()(x)
        return input_x

    return encoder


def A_HeadLayer(n_actions: int, discrete: bool) -> Callable[[tf.Tensor], Layer]:
    # n_actions = n_actions if discrete else n_actions * 2

    def head(input_x: tf.Tensor) -> tf.Tensor:
        x = Dense(32, activation="relu", kernel_initializer='he_uniform')(input_x)
        x = Dense(32, activation="tanh", kernel_initializer='he_uniform')(x)
        x = Dense(32, activation="relu", kernel_initializer='he_uniform')(x)
        if discrete:
            x = Dense(n_actions, activation="softmax")(x)
        else:
            mu = Dense(n_actions, activation="linear")(x)
            sigma = Dense(n_actions, activation="softplus")(x)
            x = Concatenate()([mu, sigma])
        return x

    return head


def C_HeadLayer() -> Callable[[tf.Tensor], Layer]:

    def head(input_x: tf.Tensor) -> tf.Tensor:
        x = Dense(32, activation="relu", kernel_initializer='he_uniform')(input_x)
        x = Dense(32, activation="tanh", kernel_initializer='he_uniform')(x)
        x = Dense(32, activation="relu", kernel_initializer='he_uniform')(x)
        x = Dense(1, activation='linear')(x)
        return x

    return head


def AC_MetaMemoryLayer(name: str) -> Callable[[tf.Tensor], Layer]:

    def memory(input_x: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        I have `input_x` input tensor which has `(batch, trajectory_shape)` shape.
        My goal is to preserve the LSTM hidden states across the entire batch (despite) using
        the `stateful=True` input parameter. In order to do this, I'm going to reshape the tensor
        in order to use the `batch_axis` as the `timestamps_axis`. In this way the LSTM layer will
        preserve the hidden states. Once I've done this, I have also to set the
        `return_sequences=True` input parameter, in order to obtain the (LSTM cell) output
        of each trajectory. And the end, I simply reshape the output tensor in order to obtain the
        initial shape `(batch, trajectory_shape)`.
        """
        ### input_x -> (batch, trajectory_shape)
        ### input_x -> (None, trajectory_shape)

        ### input_x -> (None, trajectory_shape)
        x = tf.expand_dims(input_x, axis=0)
        ### x -> (1, None, trajectory_shape)

        ### LSTM (input) -> (batch, timestamps, trajectory_shape)
        ### x -> (1, None, trajectory_shape)
        x, memory_state, carry_state = LSTM(
            512,
            return_state=True,
            return_sequences=True,
            name=name,
            #
            stateful=True,
            # batch_input_shape=(None, ) + input_x.shape,
        )(x)
        ### LSTM (out) -> (batch, timestamps, trajectory_shape)
        ### x -> (1, None, trajectory_shape)

        ### x -> (1, None, trajectory_shape)
        x = tf.squeeze(x, axis=0)
        ### x -> (None, trajectory_shape)

        return x, [memory_state, carry_state]

    return memory
