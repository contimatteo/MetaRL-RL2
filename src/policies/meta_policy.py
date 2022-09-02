from typing import Any

import abc
import numpy as np

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
