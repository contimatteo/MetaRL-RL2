# pylint: disable=wrong-import-order,wrong-import-position

# from dotenv import load_dotenv
# load_dotenv()

###

import warnings
# warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import time
import tensorflow

time.sleep(2.)
os.system('clear')

# ###

# import os
# # import logging
# # logging.disable(logging.WARNING)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# # logging.getLogger("tensorflow").setLevel(logging.ERROR)
# import tensorflow as tf
# import tensorflow.python.util.deprecation as deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
# # # tf.get_logger().setLevel('INFO')
# # # tf.debugging.set_log_device_placement(False)

# ###

import gym

gym.logger.set_level(gym.logger.ERROR)
