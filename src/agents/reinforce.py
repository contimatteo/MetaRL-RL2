# from typing import Any
# import numpy as np
# import tensorflow as tf
# from tensorflow.python.keras import Model, Sequential
# from agents.baseline import Agent

import gym

from memory import SequentialMemory
from __types import T_Action, T_State, T_Reward

# ###

# class REINFORCE(Agent):
#     def __init__(self) -> None:
#         super().__init__()
#         self.policy_network = None
#     #
#     def _policy_network(self) -> Model:
#         return Sequential()
#     #
#     def initialize(self) -> None:
#         self.policy_network = self._policy_network()
#     #
#     def train(self):
#         pass
#     def test(self):
#         pass


class REINFORCE():

    def __init__(self, n_max_episode_steps: int) -> None:
        self.memory = SequentialMemory(n_max_episode_steps)

    def act(self, env: gym.Env, state: T_State) -> T_Action:
        return env.action_space.sample()

    def remember(
        self, step: int, state: T_State, action: T_Action, reward: T_Reward, next_state: T_State
    ) -> None:
        self.memory.store(
            step, {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state
            }
        )