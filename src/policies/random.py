from typing import Any

import numpy as np

from .policy import Policy

###


class RandomPolicy(Policy):

    def _act(self, trajectory: Any, **kwargs) -> np.ndarray:
        n_batches = trajectory.shape[0]
        return np.array([self.action_space.sample() for _ in range(n_batches)])
