from typing import Any

from tensorflow.python.keras.layers import Dense, Flatten

from .network import Network

###


class ValueNetwork(Network):

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

        ### head
        # self.flat = Flatten()
        ### backbone
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(512, activation='relu')
        ### output
        self.out = Dense(1, activation='linear')

    #

    def call(self, inputs: Any, training: bool = None, mask=None):
        # x = self.flat(inputs)
        x = self.d1(inputs)
        x = self.d2(x)
        x = self.out(x)
        return x
