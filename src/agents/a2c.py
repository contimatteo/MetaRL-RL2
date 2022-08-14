from typing import Any, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.keras import Model
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.optimizers import rmsprop_v2

from policies import Policy
from utils import AdvantageEstimateUtils

from .agent import Agent

###

T_Tensor = tf.Tensor
T_TensorsTuple = Tuple[T_Tensor, T_Tensor]

###


class A2C(Agent):

    PY_NUMERIC_EPS = 1e-8

    def __init__(
        self,
        n_max_episode_steps: int,
        policy: Policy,
        actor_network: Model,
        critic_network: Model,
        gamma: float = 0.99,
        standardize_advantage_estimate: bool = False,
        critic_loss_coef: float = 0.5,
        opt_gradient_clip_norm: Optional[float] = None,  # 0.25,
        opt_actor_lr: float = 5e-5,
        opt_critic_lr: float = 5e-5,
    ) -> None:
        super(A2C, self).__init__(n_max_episode_steps=n_max_episode_steps, policy=policy)

        self._gamma = gamma
        self._critic_loss_coef = critic_loss_coef
        self._opt_gradient_clip_norm = opt_gradient_clip_norm
        self._standardize_advantage_estimate = standardize_advantage_estimate

        self.actor_network = actor_network
        self.critic_network = critic_network

        # self.actor_network_optimizer = adam_v2.Adam(learning_rate=opt_actor_lr)
        self.actor_network_optimizer = rmsprop_v2.RMSprop(learning_rate=1e-4)
        # self.critic_network_optimizer = adam_v2.Adam(learning_rate=opt_critic_lr)
        self.critic_network_optimizer = rmsprop_v2.RMSprop(learning_rate=1e-4)

    #

    @property
    def name(self) -> str:
        return "A2C"

    def _discount_rewards(self, rewards: np.ndarray) -> np.ndarray:
        # discounted_rewards = rewards
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        discounted_rewards, reward_sum = [], 0
        rewards = rewards.tolist()
        rewards.reverse()
        for r in rewards:
            reward_sum = r + self._gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        return tf.cast(discounted_rewards, tf.float32)

    #

    def _advantage_estimates(
        self, rewards: np.ndarray, state_v: np.ndarray, next_state_v: np.ndarray, dones: T_Tensor
    ) -> T_Tensor:
        ### TODO: we have to use the "N-Step Advantage Estimate"
        # return tf.stop_gradient(AdvantageEstimateUtils.MC(rewards, state_v))
        # return tf.stop_gradient(
        #     AdvantageEstimateUtils.TD(self._gamma, rewards, state_v, next_state_v, dones)
        # )
        disc_rewards = self._discount_rewards(rewards)
        return tf.stop_gradient(AdvantageEstimateUtils.MC(disc_rewards, state_v))

    def _standardize_advantages(self, advantages: Any) -> Any:
        if self._standardize_advantage_estimate:
            return AdvantageEstimateUtils.standardize(advantages)
        return advantages

    def _critic_network_loss(self, rewards: Any, advantages: Any, state_value: Any):
        # return self._critic_loss_coef * mean_squared_error(advantages, state_value)
        disc_rewards = self._discount_rewards(rewards)
        return self._critic_loss_coef * mean_squared_error(disc_rewards, state_value)

    def _actor_network_loss(self, actions_probs: Any, actions: Any, advantages: Any):
        policy_losses = []
        for step_actions_probs, action_taken, advantage_estimate in zip(
            actions_probs, actions, advantages.numpy()
        ):
            advantage = tf.constant(advantage_estimate)  ### exclude from gradient computation
            distribution = tfp.distributions.Categorical(
                probs=step_actions_probs + .000001, dtype=tf.float32
            )
            action_log_prob = distribution.log_prob(action_taken)
            policy_loss = tf.math.multiply(action_log_prob, advantage)
            policy_losses.append(policy_loss)
        policy_loss = tf.reduce_mean(tf.stack(policy_losses))
        # policy_loss = tf.reduce_sum(tf.stack(policy_losses))
        return -policy_loss

    def _clip_gradients_norm(self, a_grads: tf.Tensor, c_grads: tf.Tensor) -> T_TensorsTuple:
        if self._opt_gradient_clip_norm is not None:
            a_grads, _ = tf.clip_by_global_norm(a_grads, self._opt_gradient_clip_norm)
            c_grads, _ = tf.clip_by_global_norm(c_grads, self._opt_gradient_clip_norm)

        return a_grads, c_grads

    #

    def act(self, state: np.ndarray) -> np.ndarray:
        return self.policy.act(state)

    def train(self, batch_size: int) -> Any:
        ep_data = self.memory.all()

        states = ep_data["states"]
        rewards = ep_data["rewards"]
        actions = ep_data["actions"]
        next_states = ep_data["next_states"]
        _dones = ep_data["done"]

        assert states.shape[0] == rewards.shape[0] == actions.shape[0]
        assert states.shape[0] == next_states.shape[0] == _dones.shape[0]

        dones = tf.cast(tf.cast(_dones, tf.int8), tf.float32)

        #

        with tf.GradientTape() as a_tape, tf.GradientTape() as c_tape:
            actions_probs = self.actor_network(states, training=True)
            states_val = self.critic_network(states, training=True)
            next_states_val = self.critic_network(next_states, training=True)

            states_val = tf.reshape(states_val, (len(states_val)))
            next_states_val = tf.reshape(next_states_val, (len(next_states_val)))

            ### Action Advantage Estimates
            advantages = tf.stop_gradient(
                self._advantage_estimates(rewards, states_val, next_states_val, dones)
            )
            advantages = tf.stop_gradient(self._standardize_advantages(advantages))

            actor_loss = self._actor_network_loss(actions_probs, actions, advantages)
            critic_loss = self._critic_network_loss(rewards, advantages, states_val)

            assert not tf.math.is_inf(actor_loss) and not tf.math.is_nan(actor_loss)
            assert not tf.math.is_inf(critic_loss) and not tf.math.is_nan(critic_loss)

        actor_grads = a_tape.gradient(actor_loss, self.actor_network.trainable_variables)
        critic_grads = c_tape.gradient(critic_loss, self.critic_network.trainable_variables)

        actor_grads, critic_grads = self._clip_gradients_norm(actor_grads, critic_grads)

        self.actor_network_optimizer.apply_gradients(
            zip(actor_grads, self.actor_network.trainable_variables)
        )
        self.critic_network_optimizer.apply_gradients(
            zip(critic_grads, self.critic_network.trainable_variables)
        )

        #

        return actor_loss, critic_loss
