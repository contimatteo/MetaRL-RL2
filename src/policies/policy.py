from typing import Any

import abc
import gym
import numpy as np

from gym.spaces import Discrete

###


class Policy(abc.ABC):

    def __init__(self, state_space: gym.Space, action_space: gym.Space, action_buonds: list = None):
        self.state_space = state_space
        self.action_space = action_space

        if not self._is_discrete:
            assert isinstance(action_buonds, list)
            assert len(action_buonds) == 2

        self.action_bounds = action_buonds

    #

    @abc.abstractmethod
    def _act(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @property
    def _is_discrete(self) -> bool:
        return isinstance(self.action_space, Discrete)

    def _clip_continuous_actions(self, actions):
        return np.clip(actions, self.action_bounds[0], self.action_bounds[1])

    #

    # def log_probability(self, output, action):
    #     raise NotImplementedError

    # def entropy(self, output):
    #     raise NotImplementedError

    #

    def act(self, obs: np.ndarray, mask=None) -> np.ndarray:
        assert isinstance(obs, np.ndarray)

        if len(obs.shape) < 2:
            ### reshape in order to match network `batch` dimension
            obs = np.expand_dims(obs, axis=0)  ### (x,) -> (1, x)

        assert len(obs.shape) > 0  ### batch dimension is required

        actions = self._act(obs, mask=mask)

        if not self._is_discrete:
            actions = self._clip_continuous_actions(actions)

        return actions
