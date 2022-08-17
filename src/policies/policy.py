from typing import Optional, Any

import abc
import gym
import numpy as np

from utils import ActionUtils

###


class Policy(abc.ABC):

    def __init__(
        self,
        state_space: gym.Space,
        action_space: gym.Space,
        action_buonds: Optional[list] = None
    ):
        self.state_space = state_space
        self.action_space = action_space

        if not self._is_discrete:
            assert isinstance(action_buonds, list)
            assert len(action_buonds) == 2

        self.action_bounds = action_buonds

    #

    @abc.abstractmethod
    def _act(self, trajectory: Any, **kwargs) -> np.ndarray:
        pass

    @property
    def _is_discrete(self) -> bool:
        return ActionUtils.is_space_discrete(self.action_space)

    #

    def _add_batch_dim_to_trajectory(self, trajectory: np.ndarray) -> None:
        assert isinstance(trajectory, np.ndarray)
        assert len(trajectory.shape) == 1

        ### reshape in order to match network `batch` dimension
        trajectory = np.expand_dims(trajectory, axis=0)  ### (x,) -> (1, x)

        assert len(trajectory.shape) > 1  ### batch dimension is required

        return trajectory

    #

    def act(self, trajectory: np.ndarray, mask=None) -> np.ndarray:
        assert isinstance(trajectory, np.ndarray) or isinstance(trajectory, list)

        # if len(obs.shape) < 2:
        #     ### reshape in order to match network `batch` dimension
        #     obs = np.expand_dims(obs, axis=0)  ### (x,) -> (1, x)

        # assert len(obs.shape) > 0  ### batch dimension is required
        # assert obs.shape[0] == 1

        trajectory = self._add_batch_dim_to_trajectory(trajectory)

        actions = self._act(trajectory, mask=mask)

        if self._is_discrete:
            actions = actions.astype(int)
        if not self._is_discrete:
            actions = ActionUtils.clip_values(actions, self.action_bounds)

        return actions
