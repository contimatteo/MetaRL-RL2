from typing import Union, Optional

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from gym.spaces import Discrete
from gym.spaces import Box
from tensorflow_probability.python.distributions import Distribution

###


class ActionUtils():

    @staticmethod
    def is_space_discrete(space: Union[Discrete, Box]) -> bool:
        return isinstance(space, Discrete)

    @staticmethod
    def n(space: Union[Discrete, Box]) -> bool:
        if ActionUtils.is_space_discrete(space):
            return space.n
        else:
            return space.shape[0]

    @staticmethod
    def coefficients_to_distribution(
        space: Union[Discrete, Box], coefficients: tf.Tensor
    ) -> Distribution:
        discrete = ActionUtils.is_space_discrete(space)
        n_actions = ActionUtils.n(space)

        assert len(coefficients.shape) == 1

        if discrete:
            assert coefficients.shape[0] == n_actions

            probabilities = coefficients
            distribution = tfp.distributions.Categorical(logits=probabilities, dtype=tf.float32)
        else:
            assert coefficients.shape[0] == n_actions * 2

            mu = coefficients[:n_actions]
            sigma = coefficients[n_actions:]
            # sigma = tf.nn.softplus(sigma)
            sigma += 1e-5

            assert mu.shape[0] == sigma.shape[0] == n_actions

            distribution = tfp.distributions.Normal(loc=mu, scale=sigma, validate_args=True)
            # distribution = tfp.distributions.MultivariateNormalDiag(
            #     loc=mu, scale=sigma, validate_args=True
            # )
            # distribution = tfp.distributions.TruncatedNormal(
            #     loc=mu, scale=sigma, low=bounds[0], high=bounds[1]
            # )

        return distribution

    @staticmethod
    def clip_values(actions, bounds: list) -> np.ndarray:
        return np.clip(actions, bounds[0], bounds[1])
