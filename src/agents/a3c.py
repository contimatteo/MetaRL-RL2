from typing import Any, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.losses import MeanSquaredError

from utils import AdvantageEstimateUtils

from .a2c import A2C

###

T_Tensor = tf.Tensor
T_TensorsTuple = Tuple[T_Tensor, T_Tensor]

###


class A3C(A2C):
    """
    Advantage Actor-Critic (A2C)
    TODO:
     - {entropy_loss_coef} theory + application
     - {critic_loss_coef} theory + application
     - {_action_advantage_estimate} must be rewritten following "N-Step Advantage Estimate"
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
        # gamma = self._gamma
        # gae_lambda = self._gae_lambda
        # rewards = rewards.numpy()
        # not_dones = 1 - np.reshape(done, -1)
        # advantages = np.zeros_like(state_v)
        # advantages[-1] = rewards[-1] + (gamma * not_dones[-1] + next_state_v[-1]) - state_v[-1]
        # for t in reversed(range(len(rewards) - 1)):
        #     delta = rewards[t] + (gamma * not_dones[t] * next_state_v[t]) - state_v[t]
        #     advantages[t] = delta + (gamma * gae_lambda * advantages[t + 1] * not_dones[t])
        # # returns = tf.convert_to_tensor(advantages + state_v, dtype=tf.float32)
        # advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        # ### TODO: with or without this tensor?
        # # return tf.stop_gradient(returns), tf.stop_gradient(advantages)
        # return tf.stop_gradient(advantages)

        _rewards = tf.cast(rewards, tf.float32)

        advantages = AdvantageEstimateUtils.GAE(
            self._gamma, self._gae_lambda, _rewards, state_v, next_state_v, dones
        )

        return tf.stop_gradient(advantages)

    #

    def _actor_network_loss(self, actions_probs: Any, actions: Any, advantages: Any):
        entropy_losses = []
        policy_losses = []

        for probs, action_taken, advantage in zip(actions_probs, actions, advantages.numpy()):
            advantage = tf.constant(advantage)  ### exclude from gradient computation
            distribution = tfp.distributions.Categorical(probs=probs + .000001, dtype=tf.float32)
            #
            action_prob = distribution.prob(action_taken)
            action_log_prob = distribution.log_prob(action_taken)
            ### Policy
            policy_loss = tf.math.multiply(action_log_prob, advantage)
            policy_losses.append(policy_loss)
            ### Entropy
            entropy_loss = tf.math.negative(tf.math.multiply(action_prob, action_log_prob))
            entropy_losses.append(entropy_loss)

        def __batch_loss_reduction(batch_losses):
            loss = tf.stack(batch_losses)
            ### TODO: which is the right loss reduce operator?
            # return tf.reduce_mean(loss)
            return tf.reduce_sum(loss)

        policy_loss = __batch_loss_reduction(policy_losses)
        entropy_loss = __batch_loss_reduction(entropy_losses)

        entropy_loss = tf.math.multiply(self._entropy_loss_coef, entropy_loss)

        ### the Actor loss is the "negative log-likelihood"
        return -policy_loss - entropy_loss

    def _critic_network_loss(
        self, rewards: Any, disc_rewards: tf.Tensor, advantages: Any, state_value: Any
    ):
        # loss = mean_squared_error(advantages, state_value)
        loss = MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)(advantages, state_value)

        return self._critic_loss_coef * loss
