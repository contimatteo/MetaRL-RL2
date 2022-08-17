# pylint: disable=wrong-import-order,wrong-import-position

from dotenv import load_dotenv

load_dotenv()

###

import os
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

###

import logging
# logging.getLogger("tensorflow").setLevel(logging.WARNING)

import tensorflow as tf
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
# tf.get_logger().setLevel('INFO')

# tf.debugging.set_log_device_placement(False)

###

import gym

gym.logger.set_level(gym.logger.ERROR)
# gym.logger.set_level(gym.logger.DISABLED)
