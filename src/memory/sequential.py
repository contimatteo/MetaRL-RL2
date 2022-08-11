import tensorflow as tf
import numpy as np

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
        # return self._memory
        return {
            "steps": self._memory["steps"],
            "states": np.array(self._memory["states"]),
            "rewards": np.array(self._memory["rewards"]),
            "actions": np.array(self._memory["actions"]),
            "next_states": np.array(self._memory["next_states"]),
            "done": np.array(self._memory["done"]),
        }

    def batches(self, batch_size: int):
        batches = []

        for i in range(0, len(self._memory["states"]), batch_size):
            batches.append(
                {
                    "steps": batch_size,
                    "states": self._memory["states"][i:i + batch_size],
                    "rewards": self._memory["rewards"][i:i + batch_size],
                    "actions": self._memory["actions"][i:i + batch_size],
                    "next_states": self._memory["next_states"][i:i + batch_size],
                    "done": self._memory["done"][i:i + batch_size],
                }
            )

        return batches

    def to_tf_dataset(self) -> tf.data.Dataset:
        # return tf.data.Dataset.from_tensor_slices(
        #     np.column_stack(
        #         (
        #             np.array(self._memory["states"]),
        #             np.array(self._memory["rewards"]),
        #             np.array(self._memory["actions"]),
        #             np.array(self._memory["next_states"]),
        #             np.array(self._memory["done"]),
        #         )
        #     )
        # )

        return tf.data.Dataset.from_tensor_slices(
            {
                "states": np.array(self._memory["states"]),
                "rewards": np.array(self._memory["rewards"]),
                "actions": np.array(self._memory["actions"]),
                "next_states": np.array(self._memory["next_states"]),
                "done": np.array(self._memory["done"]),
            }
        )
