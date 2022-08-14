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
            assert isinstance(_input,
                              np.ndarray) or (isinstance(_input, float) or isinstance(_input, int))

            if isinstance(_input, np.ndarray):
                if len(_input.shape) == 1:
                    ### reshape in order to match network `batch` dimension
                    trajectory[i] = np.expand_dims(_input, axis=0)  ### (x,) -> (1, x)
            else:
                ### reshape in order to match network `batch` dimension
                trajectory[i] = np.expand_dims(_input, axis=0)  ### (x,) -> (1, x)

        for i in range(len(trajectory) - 1):
            ### batch dimension is required
            assert len(trajectory[i].shape) > 0
            ### multi-input must have the same batch_dimension
            assert trajectory[i].shape[0] == trajectory[i + 1].shape[0]

        return self._act(trajectory, mask=mask)
