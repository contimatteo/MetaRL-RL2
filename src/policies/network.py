from typing import Any
from enum import Enum

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.keras import Model

from .policy import Policy

###

ACT_MODES = ['distribution', 'argmax']

###


class NetworkPolicy(Policy):

    def __init__(
        self,
        state_space: gym.Space,
        action_space: gym.Space,
        network: Model,
        action_sampling_mode: str = 'distribution'
    ):
        super().__init__(state_space, action_space)

        assert action_sampling_mode in ACT_MODES

        self.policy_network = network
        self.action_sampling_mode = action_sampling_mode

    def _act(self, obs: Any, **kwargs) -> np.ndarray:
        assert len(obs.shape) > 1  ### batch dimension is required

        act_probs = self.policy_network(obs, training=False).numpy()

        if self.action_sampling_mode == "distribution":
            distribution = tfp.distributions.Categorical(
                probs=act_probs + .000001, dtype=tf.float32
            )

            actions = distribution.sample().numpy().astype(int)
            assert actions.shape[0] == obs.shape[0]

            return actions

        raise Exception("[DeepNetworkPolicy] `action_sampling_mode` not supported.")
