import tensorflow as tf

###


class SequentialMemory():

    def __init__(self, n_max_steps: int) -> None:
        self._memory = None

        self.n_max_steps = n_max_steps

        self.reset()

    #

    def reset(self):
        self._memory = {}
        self._memory["steps"] = 0
        self._memory["states"] = []
        self._memory["rewards"] = []
        self._memory["actions"] = []
        self._memory["next_states"] = []
        self._memory["done"] = []

    def store(self, step: int, data: dict) -> None:
        assert isinstance(step, int)
        assert isinstance(data, dict)

        self._memory["steps"] += 1
        self._memory["states"].append(data["state"])
        self._memory["rewards"].append(data["reward"])
        self._memory["actions"].append(data["action"])
        self._memory["next_states"].append(data["next_state"])
        self._memory["done"].append(data["done"])

    def all(self) -> list:
        return self._memory

    def to_dataset(self) -> tf.data.Dataset:
        return None
