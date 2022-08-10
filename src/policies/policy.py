from typing import Any

import abc
import gym

###


class Policy(abc.ABC):

    def __init__(self, state_space: gym.Space, action_space: gym.Space):
        self.state_space = state_space
        self.action_space = action_space

    #

    @abc.abstractmethod
    def _act(self, obs, **kwargs) -> Any:
        pass

    #

    # def log_probability(self, output, action):
    #     raise NotImplementedError

    # def entropy(self, output):
    #     raise NotImplementedError

    #

    def act(self, obs, mask=None, training=True) -> Any:
        return self._act(obs, mask=mask, training=training)
