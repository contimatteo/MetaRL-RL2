from typing import Any

import itertools
import gym
import numpy as np

from memory import SequentialMemory

from __types import T_Action, T_State, T_Reward

###


class Agent():

    def __init__(self, env_name: str, n_max_episode_steps: int) -> None:
        self.memory = SequentialMemory(n_max_episode_steps)

        self.env = gym.make(env_name)

    #

    @property
    def _actions_space(self) -> int:
        return self.env.action_space.n

    @property
    def _state_space(self) -> tuple:
        return self.env.observation_space.shape

    def _actions(self, space: gym.Space) -> Any:
        ### https://stackoverflow.com/a/67929513
        types = [
            gym.spaces.discrete.Discrete,
            # gym.spaces.multi_binary.MultiBinary,
            # gym.spaces.multi_discrete.MultiDiscrete,
            # gym.spaces.dict.Dict,
            # gym.spaces.tuple.Tuple,
        ]
        assert type(space) in types

        #

        if isinstance(space, gym.spaces.discrete.Discrete):
            return list(range(space.n))

        if isinstance(space, gym.spaces.multi_binary.MultiBinary):
            return [
                np.reshape(np.array(element), space.n)
                for element in itertools.product(*[range(2)] * np.prod(space.n))
            ]

        if isinstance(space, gym.spaces.multi_discrete.MultiDiscrete):
            return [
                np.array(element) for element in itertools.product(*[range(n) for n in space.nvec])
            ]

        if isinstance(space, gym.spaces.dict.Dict):
            keys = space.spaces.keys()
            values_list = itertools.product(
                *[self._actions(sub_space) for sub_space in space.spaces.values()]
            )
            return [{key: value for key, value in zip(keys, values)} for values in values_list]
            # return space_list

        if isinstance(space, gym.spaces.tuple.Tuple):
            return [
                list(element) for element in
                itertools.product(*[self._actions(sub_space) for sub_space in space.spaces])
            ]

    #

    def remember(
        self, step: int, state: T_State, action: T_Action, reward: T_Reward, next_state: T_State,
        done: bool
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

    #

    def configure(self, _: T_State) -> None:
        raise NotImplementedError

    def act(self, _: T_State) -> None:
        raise NotImplementedError

    def train(self) -> None:
        raise NotImplementedError

    def test(self) -> None:
        raise NotImplementedError
