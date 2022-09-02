from typing import Any

import gym
import numpy as np

from memory import SequentialMemory
from policies import Policy
from utils import ActionUtils

###


class Agent():

    def __init__(self, n_max_episode_steps: int, policy: Policy) -> None:
        self.env = None

        self.policy = policy
        self.memory = SequentialMemory(n_max_episode_steps)

    #

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space

    @property
    def obs_space(self) -> gym.Space:
        return self.env.observation_space

    @property
    def n_actions(self) -> int:
        n_actions = self.action_space.n
        assert isinstance(n_actions, int) and n_actions > 0
        return n_actions

    @property
    def _discrete_action_space(self) -> bool:
        return ActionUtils.is_space_discrete(self.action_space)

    #

    def env_sync(self, env: gym.Env) -> None:
        assert isinstance(env, gym.Env)
        self.env = env

    def memory_reset(self) -> None:
        self.memory.reset()

    def remember(
        self, step: int, state: Any, action: Any, reward: Any, next_state: Any, done: bool
    ) -> None:

        assert isinstance(step, int)
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, float) or isinstance(reward, int)
        assert isinstance(next_state, np.ndarray)
        assert isinstance(done, bool)

        record = {}
        record["state"] = state
        record["action"] = action
        record["reward"] = float(reward)
        record["next_state"] = next_state
        record["done"] = int(done)

        self.memory.store(step, record)

    #

    def train(self, batch_size: int) -> Any:
        raise NotImplementedError
