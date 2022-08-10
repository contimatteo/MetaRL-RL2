from typing import Any

import gym

from tensorflow.python.keras.layers import Dense, Flatten

from .network import Network

###


class PolicyNetwork(Network):

    def __init__(self, action_space: gym.Space, *args: Any, **kwargs: Any):
        super().__init__()

        ### TODO: support also 'continuous' action space
        assert isinstance(action_space, gym.spaces.discrete.Discrete)

        self._action_space = action_space

        ### head
        self.flat = Flatten()
        ### backbone
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(512, activation='relu')
        ### output
        self.out = Dense(action_space.n, activation='softmax' if self.discrete else 'linear')

    @property
    def discrete(self) -> bool:
        return isinstance(self._action_space, gym.spaces.discrete.Discrete)

    #

    def call(self, inputs: Any, training: bool = None, mask=None):
        x = self.flat(inputs)
        x = self.d1(x)
        x = self.d2(x)
        x = self.out(x)
        return x
