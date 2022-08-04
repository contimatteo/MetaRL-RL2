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
        self._memory = np.full((self.n_max_steps), {})

    #

    def store(self, step: int, data: dict) -> None:
        assert isinstance(step, int)
        assert isinstance(data, dict)

        # self._memory[step] = json.dumps(data)
        self._memory[step] = data

    def reset(self) -> None:
        self.__reset_memory()
