from typing import Any

import gym

from memory import SequentialMemory

###


class MetaAgent():

    def __init__(self, n_max_episode_steps: int) -> None:
        self.env = None

        self.memory = SequentialMemory(n_max_episode_steps)

    #

    @property
    def name(self) -> str:
        return "MetaAgent"

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
        episode_metrics = []
        ### ...
        return episode_metrics

    def test(self):
        pass
