from typing import Any

import numpy as np

from .meta_policy import MetaPolicy

###


class RandomMetaPolicy(MetaPolicy):

    def _act(self, trajectory: Any, **kwargs) -> np.ndarray:
        n_batches = trajectory[0].shape[0]
        return np.array([self.action_space.sample() for _ in range(n_batches)])
