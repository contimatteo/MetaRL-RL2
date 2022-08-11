from typing import Any

import numpy as np
import tensorflow as tf

from .a2c import A2C

###


class MetaA2C(A2C):

    @property
    def name(self) -> str:
        return "MetaA2C"

    #

    def train(self, batch_size: int) -> Any:
        raise NotImplementedError
