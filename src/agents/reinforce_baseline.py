# from typing import Any

# import numpy as np
# import tensorflow as tf

# from tensorflow.python.keras import Model, Sequential

# from algorithms.base import StandardAlgorithm

# ###

# class REINFORCEWithBaseline(REINFORCE):

#     def __init__(self) -> None:
#         super().__init__()

#         self.value_network = self._build_value_network()
#         self.policy_network = self._build_policy_network()

#     #

#     def _build_value_network(self) -> Model:
#         return Sequential()

#     def _build_policy_network(self) -> Model:
#         return Sequential()

#     def _compute_baseline_value(self, state: tf.Tensor) -> np.ndarray:
#         return self.value_network.predict(state)

#     #

#     def initialize(self) -> None:
#         self.value_network = self._build_value_network()
#         self.policy_network = self._build_policy_network()
