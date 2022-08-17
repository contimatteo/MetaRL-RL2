from typing import Any

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.keras import Model

from utils import ActionUtils
from .meta_policy import MetaPolicy

###

ACT_MODES = ['distribution', 'argmax']

###


class NetworkMetaPolicy(MetaPolicy):

    def __init__(
        self,
        state_space: gym.Space,
        action_space: gym.Space,
        network: Model,
        action_buonds: list = None,
        action_sampling_mode: str = 'distribution'
    ):
        super().__init__(state_space, action_space, action_buonds)

        assert action_sampling_mode in ACT_MODES

        self.policy_network = network
        self.action_sampling_mode = action_sampling_mode

    def _act(self, trajectory: Any, **kwargs) -> np.ndarray:
        coefficients = self.policy_network(trajectory, training=False)

        ### match batch dimension
        assert coefficients.shape[0] == trajectory[0].shape[0]
        ### remove batch dimension
        coefficients = coefficients[0]

        distribution = ActionUtils.coefficients_to_distribution(self.action_space, coefficients)

        actions = distribution.sample()
        actions = actions.numpy()

        return actions

        # _actions = self.policy_network(trajectory, training=False)

        # if not self._is_discrete:
        #     return _actions

        # act_probs = _actions.numpy()

        # if self.action_sampling_mode == "distribution":
        #     # distribution = tfp.distributions.Categorical(
        #     #     probs=act_probs + .000001, dtype=tf.float32
        #     # )
        #     distribution = tfp.distributions.Categorical(logits=act_probs, dtype=tf.float32)

        #     actions = distribution.sample().numpy()
        #     assert actions.shape[0] == trajectory[0].shape[0]

        #     return actions

        # raise Exception("[DeepNetworkPolicy] `action_sampling_mode` not supported.")
