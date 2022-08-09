from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense

###


class CriticNetwork(Model):

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

        self.d1 = Dense(512, activation='relu')
        self.out = Dense(1, activation='linear')

    def call(self, input_data: Any, training=None, mask=None):
        x = self.d1(input_data)
        x = self.out(x)
        return x


class ActorNetwork(Model):

    def __init__(self, n_actions: int, *args: Any, **kwargs: Any):
        super().__init__()

        assert isinstance(n_actions, int)

        self.d1 = Dense(512, activation='relu')
        self.out = Dense(n_actions, activation='softmax')  # discrete
        # self.out = Dense(n_actions, activation='linear')  # continuous

    def call(self, input_data: Any, training=None, mask=None):
        x = self.d1(input_data)
        x = self.out(x)
        return x
