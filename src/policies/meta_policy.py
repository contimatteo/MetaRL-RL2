from typing import Any

import abc
import gym
import numpy as np

from utils import ActionUtils
from .policy import Policy

###


class MetaPolicy(Policy, abc.ABC):

    def _add_batch_dim_to_trajectory(self, trajectory: Any) -> None:
        assert isinstance(trajectory, list)
        assert len(trajectory[0].shape) == 1

        for i, _input in enumerate(trajectory):
            _input = np.array(_input, dtype=np.float32)

            if len(_input.shape) < 2:
                ### reshape in order to match network `batch` dimension
                _input = np.expand_dims(_input, axis=0)  ### (x,) -> (1, x)

            assert len(_input.shape
                       ) > len(np.array(trajectory[i]).shape)  ### batch dimension is required
            trajectory[i] = _input

        return trajectory

    #

    # def act(self, trajectory: list, mask=None) -> np.ndarray:
    #     assert isinstance(trajectory, list)

    #     assert trajectory[0].shape[0] == 1

    #     for i, _input in enumerate(trajectory):
    #         _input = np.array(_input, dtype=np.float32)

    #         if len(_input.shape) < 2:
    #             ### reshape in order to match network `batch` dimension
    #             _input = np.expand_dims(_input, axis=0)  ### (x,) -> (1, x)

    #         trajectory[i] = _input
    #         assert len(trajectory[i].shape) > 0  ### batch dimension is required

    #     actions = self._act(trajectory, mask=mask)

    #     if self._is_discrete:
    #         actions = actions.astype(int)
    #     if not self._is_discrete:
    #         actions = ActionUtils.clip_values(actions, self.action_bounds)

    #     return actions
