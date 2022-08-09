# pylint: disable=wrong-import-order,wrong-import-position

import os
import warnings

from dotenv import load_dotenv

load_dotenv()

###

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

###

import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
# logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

###

import gym

gym.logger.set_level(gym.logger.ERROR)
# gym.logger.set_level(gym.logger.DISABLED)
