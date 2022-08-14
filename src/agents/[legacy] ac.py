# from typing import Any, Optional, Tuple

# import numpy as np
# import tensorflow as tf

# from tensorflow.python.keras import Model
# from tensorflow.python.keras.optimizers import Optimizer

# from policies import Policy
# from utils import AdvantageEstimateUtils

# from .agent import Agent

# ###

# T_Tensor = tf.Tensor
# T_TensorsTuple = Tuple[T_Tensor, T_Tensor]

# ###

# class AC(Agent):

#     def __init__(
#         self,
#         n_max_episode_steps: int,
#         policy: Policy,
#         actor_network: Model,
#         critic_network: Model,
#         actor_network_opt: Optimizer,
#         critic_network_opt: Optimizer,
#         opt_gradient_clip_norm: Optional[float] = 999.0,
#         gamma: float = 0.99,
#         gae_lambda: float = 0.9,
#         critic_loss_coef: float = 0.5,
#         entropy_loss_coef: float = 1e-3,
#         standardize_advantage_estimate: bool = False,
#     ) -> None:
#         super(AC, self).__init__(n_max_episode_steps=n_max_episode_steps, policy=policy)

#         self._gamma = gamma
#         self._gae_lambda = gae_lambda
#         self._critic_loss_coef = critic_loss_coef
#         self._entropy_loss_coef = entropy_loss_coef
#         self._opt_gradient_clip_norm = opt_gradient_clip_norm
#         self._standardize_advantage_estimate = standardize_advantage_estimate

#         self.actor_network = actor_network
#         self.critic_network = critic_network

#         self.actor_network_optimizer = actor_network_opt
#         self.critic_network_optimizer = critic_network_opt

#     #

#     @property
#     def name(self) -> str:
#         return "AC"

#     #

#     def __discount_rewards(self, rewards: np.ndarray) -> tf.Tensor:
#         discounted_rewards, reward_sum = [], 0
#         rewards = rewards.tolist()
#         rewards.reverse()
#         for r in rewards:
#             reward_sum = r + self._gamma * reward_sum
#             discounted_rewards.append(reward_sum)
#         discounted_rewards.reverse()
#         return tf.cast(discounted_rewards, tf.float32)

#     def __standardize_advantages(self, advantages: Any) -> Any:
#         if self._standardize_advantage_estimate:
#             return AdvantageEstimateUtils.standardize(advantages)
#         return advantages

#     def __clip_gradients_norm(self, a_grads: tf.Tensor, c_grads: tf.Tensor) -> T_TensorsTuple:
#         if self._opt_gradient_clip_norm is not None:
#             a_grads, _ = tf.clip_by_global_norm(a_grads, self._opt_gradient_clip_norm)
#             c_grads, _ = tf.clip_by_global_norm(c_grads, self._opt_gradient_clip_norm)

#         return a_grads, c_grads

#     #

#     def _advantage_estimates(
#         self, rewards: np.ndarray, disc_rewards: tf.Tensor, state_v: np.ndarray,
#         next_state_v: np.ndarray, dones: T_Tensor
#     ) -> T_Tensor:
#         raise NotImplementedError

#     def _actor_network_loss(self, actions_probs: Any, actions: Any, advantages: Any):
#         raise NotImplementedError

#     def _critic_network_loss(
#         self, rewards: Any, disc_rewards: tf.Tensor, advantages: tf.Tensor, state_value: Any
#     ):
#         raise NotImplementedError

#     #

#     def act(self, state: np.ndarray) -> np.ndarray:
#         return self.policy.act(state)

#     def train(self, batch_size: Optional[int] = None, shuffle: bool = False) -> Any:
#         ep_data = self.memory.all()

#         states = ep_data["states"]
#         rewards = ep_data["rewards"]
#         actions = ep_data["actions"]
#         next_states = ep_data["next_states"]

#         assert states.shape[0] == rewards.shape[0] == actions.shape[0]
#         assert states.shape[0] == next_states.shape[0] == ep_data["done"].shape[0]

#         disc_rewards = self.__discount_rewards(rewards)
#         dones = tf.cast(tf.cast(ep_data["done"], tf.int8), tf.float32)

#         assert states.shape[0] == disc_rewards.shape[0] == dones.shape[0]

#         #

#         indexes = np.arange(states.shape[0])

#         if shuffle:
#             np.random.shuffle(indexes)
#         if batch_size is None:
#             batch_size = states.shape[0]

#         for b_start_idx in range(0, states.shape[0], batch_size):
#             b_end_idx = b_start_idx + batch_size
#             b_idxs = indexes[b_start_idx:b_end_idx]

#             _dones = tf.gather(dones, b_idxs, axis=0)
#             _states = tf.gather(states, b_idxs, axis=0)
#             _rewards = tf.gather(rewards, b_idxs, axis=0)
#             _actions = tf.gather(actions, b_idxs, axis=0)
#             _next_states = tf.gather(next_states, b_idxs, axis=0)
#             _disc_rewards = tf.gather(disc_rewards, b_idxs, axis=0)

#             with tf.GradientTape() as a_tape, tf.GradientTape() as c_tape:
#                 actions_probs = self.actor_network(_states, training=True)
#                 states_val = self.critic_network(_states, training=True)
#                 next_states_val = self.critic_network(_next_states, training=True)

#                 states_val = tf.reshape(states_val, (len(states_val)))
#                 next_states_val = tf.reshape(next_states_val, (len(next_states_val)))

#                 ### Action Advantage Estimates
#                 advantages = tf.stop_gradient(
#                     self._advantage_estimates(
#                         _rewards, _disc_rewards, states_val, next_states_val, _dones
#                     )
#                 )
#                 advantages = self.__standardize_advantages(advantages)

#                 actor_loss = self._actor_network_loss(actions_probs, _actions, advantages)
#                 critic_loss = self._critic_network_loss(
#                     _rewards, _disc_rewards, advantages, states_val
#                 )

#                 assert not tf.math.is_inf(actor_loss) and not tf.math.is_nan(actor_loss)
#                 assert not tf.math.is_inf(critic_loss) and not tf.math.is_nan(critic_loss)

#             actor_grads = a_tape.gradient(actor_loss, self.actor_network.trainable_variables)
#             critic_grads = c_tape.gradient(critic_loss, self.critic_network.trainable_variables)

#             actor_grads, critic_grads = self.__clip_gradients_norm(actor_grads, critic_grads)

#             self.actor_network_optimizer.apply_gradients(
#                 zip(actor_grads, self.actor_network.trainable_variables)
#             )
#             self.critic_network_optimizer.apply_gradients(
#                 zip(critic_grads, self.critic_network.trainable_variables)
#             )

#         #

#         return actor_loss, critic_loss
