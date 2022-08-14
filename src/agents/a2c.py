from typing import Any, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.keras.losses import mean_squared_error

from utils import AdvantageEstimateUtils

from .ac import AC

###

T_Tensor = tf.Tensor
T_TensorsTuple = Tuple[T_Tensor, T_Tensor]

###


class A2C(AC):

    @property
    def name(self) -> str:
        return "A2C"

    #

    def _advantage_estimates(
        self, rewards: np.ndarray, disc_rewards: tf.Tensor, state_v: np.ndarray,
        next_state_v: np.ndarray, dones: T_Tensor
    ) -> T_Tensor:
        ### TODO: we have to use the "N-Step Advantage Estimate"
        advantages = AdvantageEstimateUtils.MC(disc_rewards, tf.stop_gradient(state_v))
        # advantages = AdvantageEstimateUtils.TD(self._gamma, rewards, state_v, next_state_v, dones)
        # advantages = AdvantageEstimateUtils.NStep()

        return tf.stop_gradient(advantages)

    #

    def _critic_network_loss(
        self, rewards: Any, disc_rewards: tf.Tensor, advantages: Any, state_value: Any
    ):
        return self._critic_loss_coef * mean_squared_error(disc_rewards, state_value)

    def _actor_network_loss(self, actions_probs: Any, actions: Any, advantages: Any):
        policy_losses = []

        for probs, action_taken, advantage in zip(actions_probs, actions, advantages.numpy()):
            advantage = tf.constant(advantage)  ### exclude from gradient computation
            distribution = tfp.distributions.Categorical(probs=probs + .000001, dtype=tf.float32)
            action_log_prob = distribution.log_prob(action_taken)
            policy_loss = tf.math.multiply(action_log_prob, advantage)
            policy_losses.append(policy_loss)

        def __batch_loss_reduction(batch_losses):
            loss = tf.stack(batch_losses)
            ### TODO: which is the right loss reduce operator?
            # return tf.reduce_sum(loss)
            return tf.reduce_mean(loss)

        policy_loss = __batch_loss_reduction(policy_losses)

        return -policy_loss
