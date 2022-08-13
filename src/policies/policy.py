from typing import Any

import abc
import gym
import numpy as np

###


class Policy(abc.ABC):

    def __init__(self, state_space: gym.Space, action_space: gym.Space):
        self.state_space = state_space
        self.action_space = action_space

    #

    @abc.abstractmethod
    def _act(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        pass

    #

    # def log_probability(self, output, action):
    #     raise NotImplementedError

    # def entropy(self, output):
    #     raise NotImplementedError

    #

    def act(self, obs: np.ndarray, mask=None) -> np.ndarray:
        assert isinstance(obs, np.ndarray)

        if len(obs.shape) == 1:
            ### reshape in order to match network `batch` dimension
            obs = np.expand_dims(obs, axis=0)  ### (x,) -> (1, x)

        assert len(obs.shape) > 0  ### batch dimension is required

        return self._act(obs, mask=mask)
