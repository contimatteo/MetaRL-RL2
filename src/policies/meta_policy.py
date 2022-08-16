from typing import Any

import abc
import gym
import numpy as np

from .policy import Policy

###


class MetaPolicy(Policy, abc.ABC):

    #

    def act(self, trajectory: list, mask=None) -> np.ndarray:
        assert isinstance(trajectory, list)

        for i, _input in enumerate(trajectory):
            _input = np.array(_input, dtype=np.float32)

            if len(_input.shape) < 2:
                ### reshape in order to match network `batch` dimension
                _input = np.expand_dims(_input, axis=0)  ### (x,) -> (1, x)

            trajectory[i] = _input
            assert len(trajectory[i].shape) > 0  ### batch dimension is required

        actions = self._act(trajectory, mask=mask)

        if not self._is_discrete:
            actions = self._clip_continuous_actions(actions)

        return actions
