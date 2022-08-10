from typing import Any
from enum import Enum

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.keras import Model

from .policy import Policy

###

ACTION_SAMPLING_MODES = Enum('distribution', 'argmax')

###


class NetworkPolicy(Policy):

    def __init__(
        self, state_space: gym.Space, action_space: gym.Space, network: Model,
        action_sampling_mode: ACTION_SAMPLING_MODES
    ):
        super().__init__(state_space, action_space)

        assert action_sampling_mode in ACTION_SAMPLING_MODES

        self.policy_network = network
        self.action_sampling_mode = action_sampling_mode

    def _act(self, obs: Any, **kwargs) -> np.ndarray:
        actions_probabilities = self.policy_network(obs, training=False).numpy()

        if self.action_sampling_mode == "distribution":
            actions = []
            for single_action_probs in actions_probabilities:
                distribution = tfp.distributions.Categorical(
                    probs=single_action_probs + .000001, dtype=tf.float32
                )
                action_tensor = distribution.sample()
                action = int(action_tensor.numpy()[0])
                actions.append(action)
            return np.array(actions)

        raise Exception("[DeepNetworkPolicy] `action_sampling_mode` not supported.")
