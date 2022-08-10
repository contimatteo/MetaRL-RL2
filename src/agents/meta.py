from typing import Any

import gym
import numpy as np
import tensorflow as tf

from memory import SequentialMemory

###


class MetaAgent():

    def __init__(
        self, n_max_episode_steps: int, gamma: float = 0.99, gae_lambda: float = 0.9
    ) -> None:
        self.env = None

        self._gamma = gamma
        self._gae_lambda = gae_lambda

        self.memory = SequentialMemory(n_max_episode_steps)

    #

    @property
    def name(self) -> str:
        return "MetaAgent"

    #

    def _TD_error(
        self, reward: float, state_value: float, next_state_value: float, done: int
    ) -> float:
        """
        ### TD-Error (1-Step Advantage)
        `Aφ(s,a) = r(s,a,s′) + γVφ(s′) − Vφ(s)` \n
        if `s′` is terminal, then `Vφ(s′) ≐ 0`
        """
        assert isinstance(done, int)
        assert done == 0 or done == 1

        if done == 1:
            next_state_value = 0

        return reward + (self._gamma * next_state_value) - state_value

    def _generalized_advantage_estimate(self, state_values, next_state_values, rewards, dones):
        state_values = state_values.numpy()
        next_state_values = next_state_values.numpy()
        # rewards = np.reshape(rewards, -1)
        # dones = 1 - np.reshape(dones, -1)
        advantages = np.zeros_like(state_values)

        assert state_values.shape == rewards.shape

        ### advantage of the last timestep
        advantages[-1] = self._TD_error(
            rewards[-1], state_values[-1], next_state_values[-1], dones[-1]
        )

        for t in reversed(range(len(rewards) - 1)):
            delta = self._TD_error(rewards[t], state_values[t], next_state_values[t], dones[t])
            advantages[t] = delta + (self._gamma * self._gae_lambda * advantages[t + 1] * dones[t])

        returns = tf.convert_to_tensor(advantages + state_values, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        ### TODO: try with and without the {tf.stop_gradient(...)} function because, if I'm not
        ### wrong, it is useless, since we convert to `numpy` the {state_values} and
        ### {next_state_values} input tensors.

        ### INFO: stop gradient to avoid backpropagating through the advantage estimator
        # return returns, advantages
        return tf.stop_gradient(returns), tf.stop_gradient(advantages)

    #

    def sync_env(self, env: gym.Env) -> None:
        assert isinstance(env, gym.Env)
        self.env = env

    def reset_memory(self) -> None:
        self.memory.reset()

    def remember(
        self, step: int, state: Any, action: Any, reward: Any, next_state: Any, done: bool
    ) -> None:
        self.memory.store(
            step, {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
            }
        )

    def act(self, state: Any) -> Any:
        return self.env.action_space.sample()

    #

    def train(self) -> Any:
        ep_metrics = []

        #

        ep_data = self.memory.to_tf_dataset()

        for ep_data_batch in ep_data.batch(4):
            _states = ep_data_batch["states"]
            _rewards = ep_data_batch["rewards"]
            _actions = ep_data_batch["actions"]
            _next_states = ep_data_batch["next_states"]
            _done = ep_data_batch["done"]

        #

        return ep_metrics

    def test(self):
        pass
