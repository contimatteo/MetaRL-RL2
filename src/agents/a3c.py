from typing import Any, Tuple

import numpy as np
import tensorflow as tf

from tensorflow.python.keras.losses import MeanSquaredError

from utils import AdvantageEstimateUtils
from utils import ActionUtils

from .a2c import A2C

###

T_Tensor = tf.Tensor
T_TensorsTuple = Tuple[T_Tensor, T_Tensor]

###


class A3C(A2C):
    """
    Advantage Actor-Critic (A2C)
    """

    #

    @property
    def name(self) -> str:
        return "A3C"

    @property
    def meta_algorithm(self) -> bool:
        return False

    #

    def _advantage_estimates(
        self, rewards: np.ndarray, disc_rewards: tf.Tensor, state_v: np.ndarray,
        next_state_v: np.ndarray, dones: T_Tensor
    ) -> T_Tensor:
        _rewards = tf.cast(rewards, tf.float32)

        advantages, _ = AdvantageEstimateUtils.GAE(
            self._gamma, self._gae_lambda, _rewards, state_v, next_state_v, dones
        )

        return advantages

    #

    def _actor_network_loss(self, actions_coefficients: Any, actions: Any, advantages: Any):
        entropy_losses = []
        policy_losses = []

        for coeff, action_taken, advantage in zip(
            actions_coefficients, actions, advantages.numpy()
        ):
            advantage = tf.constant(advantage)  ### exclude from gradient computation
            #
            distribution = ActionUtils.coefficients_to_distribution(self.action_space, coeff)
            action_prob = distribution.prob(action_taken)
            action_log_prob = distribution.log_prob(action_taken)
            ### Policy
            policy_loss = tf.math.multiply(action_log_prob, advantage)
            policy_losses.append(policy_loss)
            ### Entropy
            entropy_loss = tf.math.multiply(action_prob, action_log_prob)
            entropy_loss = tf.math.negative(entropy_loss)
            entropy_losses.append(entropy_loss)

        def __batch_loss_reduction(batch_losses):
            loss = tf.stack(batch_losses)
            return tf.reduce_mean(loss)

        policy_loss = __batch_loss_reduction(policy_losses)
        entropy_loss = __batch_loss_reduction(entropy_losses)

        entropy_loss = tf.math.multiply(self._entropy_loss_coef, entropy_loss)

        ### the Actor loss is the "negative log-likelihood"
        return -policy_loss - entropy_loss

    def _critic_network_loss(
        self, rewards: Any, disc_rewards: tf.Tensor, advantages: Any, state_value: Any
    ):
        loss = MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)(advantages, state_value)

        return self._critic_loss_coef * loss
