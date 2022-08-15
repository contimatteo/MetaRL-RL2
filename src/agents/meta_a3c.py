from typing import Any, List, Tuple, Optional

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import LSTM

from policies import Policy

from .a3c import A3C

###


class MetaA3C(A3C):

    @property
    def name(self) -> str:
        return "MetaA3C"

    @property
    def meta_algorithm(self) -> bool:
        return True

    # @property
    # def meta_memory_layer(self) -> LSTM:
    #     memory_layer = self.memory_network.get_layer(name='MetaMemory')
    #     assert memory_layer is not None
    #     return memory_layer

    # def get_meta_memory_layer_states(self) -> List[tf.Tensor]:
    #     return self.meta_memory_layer.states

    # def set_meta_memory_layer_states(self, states: List[tf.Tensor]) -> None:
    #     assert isinstance(states, list)
    #     assert states[0] is not None and states[1] is not None
    #     # self.meta_memory_layer.reset_states(states)
    #     self.meta_memory_layer.states = states

    # def reset_memory_layer_states(self) -> None:
    #     # states = self.get_meta_memory_layer_states()
    #     # assert isinstance(states, list)
    #     # assert states[0] is not None and states[1] is not None
    #     self.meta_memory_layer.states = [None, None]

    # #

    # def _meta_trajectories(
    #     self, states: tf.Tensor, next_states: tf.Tensor, actions: tf.Tensor, rewards: tf.Tensor,
    #     prev_batch_last_action: Any, prev_batch_last_reward: Any
    # ) -> Tuple[list, list]:
    #     prev_actions = actions.numpy().copy()
    #     prev_rewards = rewards.numpy().copy()

    #     prev_actions = np.insert(prev_actions[:-1], 0, prev_batch_last_action)
    #     prev_rewards = np.insert(prev_rewards[:-1], 0, prev_batch_last_reward)

    #     # INFO: we cannot convert {actions} to one-hot (encoding) because we have also NOT Discrete spaces
    #     # prev_actions = tf.one_hot(prev_actions, self.n_actions)

    #     prev_rewards = tf.constant(prev_rewards)
    #     ### `actions` in discrete space are `int` but `keras.Model` takes only `float`
    #     prev_actions = tf.cast(prev_actions, dtype=tf.float32)

    #     assert states.shape[0] == prev_actions.shape[0] == prev_rewards.shape[0]

    #     trajectories = [states, prev_actions, prev_rewards]
    #     next_trajectories = [next_states, actions, rewards]

    #     return trajectories, next_trajectories

    #

    # def train(self, batch_size: Optional[int] = None, shuffle: bool = False) -> Any:
    #     steps_metrics = {
    #         "actor_nn_loss": [],
    #         "critic_nn_loss": [],
    #         "rewards": [],
    #     }
    #     episode_metrics = {
    #         "steps": 0,
    #         "actor_nn_loss_avg": 0,
    #         "critic_nn_loss_avg": 0,
    #         "rewards_avg": 0,
    #         "rewards_sum": 0,
    #     }

    #     ep_data = self.memory.to_tf_dataset()

    #     ## v1
    #     # ep_data_batches = ep_data.batch(batch_size, drop_remainder=True)
    #     ## v2
    #     ep_data_batches = ep_data.batch(batch_size)
    #     ep_data_batches = ep_data_batches.shuffle(ep_data_batches.cardinality()).unbatch()
    #     ep_data_batches = ep_data_batches.batch(batch_size, drop_remainder=True)

    #     #

    #     batch_index = -1
    #     prev_batch_last_action = 0
    #     prev_batch_last_reward = 0

    #     prev_meta_memory_states = self.get_meta_memory_layer_states()

    #     for ep_data_batch in ep_data_batches:
    #         batch_index += 1

    #         _states: np.ndarray = ep_data_batch["states"]
    #         _rewards: np.ndarray = ep_data_batch["rewards"]
    #         _actions: np.ndarray = ep_data_batch["actions"]
    #         _next_states: np.ndarray = ep_data_batch["next_states"]
    #         _done: np.ndarray = ep_data_batch["done"]

    #         _disc_rewards = self._discount_rewards(_rewards)
    #         _rewards = tf.cast(_rewards, tf.float32)
    #         _disc_rewards = tf.cast(_disc_rewards, tf.float32)

    #         assert _states.shape[0] == _disc_rewards.shape[0] == _actions.shape[0]
    #         assert _states.shape[0] == _next_states.shape[0] == _done.shape[0]
    #         assert batch_index < 1 or prev_meta_memory_states[0] is not None
    #         assert batch_index < 1 or prev_meta_memory_states[1] is not None

    #         meta_trajectories, meta_next_trajectories = self._meta_trajectories(
    #             _states, _next_states, _actions, _rewards, prev_batch_last_action,
    #             prev_batch_last_reward
    #         )

    #         ### assert that all elements of the trajectories have the same batch_size dimension
    #         for i in range(len(meta_trajectories) - 1):
    #             assert meta_trajectories[i].shape[0] == meta_next_trajectories[i].shape[0]
    #             assert meta_trajectories[i].shape[0] == meta_trajectories[i + 1].shape[0]
    #             assert meta_next_trajectories[i].shape[0] == meta_next_trajectories[i + 1].shape[0]

    #         prev_batch_last_action = _actions[-1]
    #         prev_batch_last_reward = _rewards[-1]

    #         #

    #         ### INFO: persist `meta-memory` layer states across batches
    #         if batch_index > 0:
    #             self.set_meta_memory_layer_states(prev_meta_memory_states)

    #         with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
    #             actions_probs = self.actor_network(meta_trajectories, training=True)
    #             state_values = self.critic_network(meta_trajectories, training=True)
    #             prev_meta_memory_states = self.memory_network(meta_trajectories, training=False)

    #             ### ISSUE: how we handle the `trajectories` for the `_next_states`?
    #             ### NB: `next_state_values` are required for computing actions `advantage_estimates`
    #             bootstrap_value = 0.  ### FIXME: how can we derive this?
    #             next_state_values = state_values + bootstrap_value
    #             # next_state_values = self.critic_network(meta_next_trajectories, training=True)

    #             state_values = tf.reshape(state_values, (len(state_values)))
    #             next_state_values = tf.reshape(next_state_values, (len(next_state_values)))

    #             action_advantages = tf.stop_gradient(
    #                 self._advantage_estimates(
    #                     _disc_rewards, state_values, next_state_values, _done
    #                 )
    #             )

    #             actor_loss = self._actor_network_loss(actions_probs, _actions, action_advantages)
    #             critic_loss = self._critic_network_loss(action_advantages, state_values)

    #         actor_grads = tape1.gradient(actor_loss, self.actor_network.trainable_variables)
    #         critic_grads = tape2.gradient(critic_loss, self.critic_network.trainable_variables)

    #         if self._opt_gradient_clip_norm is not None:
    #             actor_grads, _ = tf.clip_by_global_norm(actor_grads, self._opt_gradient_clip_norm)
    #             critic_grads, _ = tf.clip_by_global_norm(critic_grads, self._opt_gradient_clip_norm)

    #         self.actor_network_optimizer.apply_gradients(
    #             zip(actor_grads, self.actor_network.trainable_variables)
    #         )
    #         self.critic_network_optimizer.apply_gradients(
    #             zip(critic_grads, self.critic_network.trainable_variables)
    #         )

    #         steps_metrics["actor_nn_loss"].append(actor_loss)
    #         steps_metrics["critic_nn_loss"].append(critic_loss)
    #         steps_metrics["rewards"] += _rewards.numpy().tolist()

    #     episode_metrics["steps"] = ep_data.cardinality().numpy()
    #     episode_metrics["actor_nn_loss_avg"] = np.mean(steps_metrics["actor_nn_loss"])
    #     episode_metrics["critic_nn_loss_avg"] = np.mean(steps_metrics["critic_nn_loss"])
    #     episode_metrics["actor_nn_loss_sum"] = np.sum(steps_metrics["actor_nn_loss"])
    #     episode_metrics["critic_nn_loss_sum"] = np.sum(steps_metrics["critic_nn_loss"])
    #     episode_metrics["rewards_avg"] = np.mean(steps_metrics["rewards"])
    #     episode_metrics["rewards_sum"] = np.sum(steps_metrics["rewards"])

    #     return episode_metrics
