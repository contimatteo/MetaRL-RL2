import numpy as np

from .policy import Policy

###


class RandomPolicy(Policy):

    def _act(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        n_batches = obs.shape[0]
        return np.array([self.action_space.sample() for _ in range(n_batches)])
