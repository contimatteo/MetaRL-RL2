import json
import numpy as np

###


class SequentialMemory():

    def __init__(self, n_max_steps: int) -> None:
        self._memory = None

        self.n_max_steps = n_max_steps

        self.__reset_memory()

    #

    def __reset_memory(self):
        # self._memory = [None for _ in range(self.n_max_steps)]

        self._memory = {}
        self._memory["steps"] = 0
        self._memory["states"] = []
        self._memory["rewards"] = []
        self._memory["actions"] = []
        self._memory["next_states"] = []

    #

    def reset(self) -> None:
        self.__reset_memory()

    def all(self) -> list:
        return self._memory

    def store(self, step: int, data: dict) -> None:
        assert isinstance(step, int)
        assert isinstance(data, dict)

        self._memory["steps"] += 1
        self._memory["states"].append(data["state"])
        self._memory["rewards"].append(data["reward"])
        self._memory["actions"].append(data["action"])
        self._memory["next_states"].append(data["next_state"])
