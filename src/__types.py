from typing import Union, List

import numpy as np

###

T_Action = Union[int, np.ndarray]
T_Actions = List[T_Action]

T_State = np.ndarray
T_States = np.ndarray

T_Reward = float
T_Rewards = List[T_Reward]
