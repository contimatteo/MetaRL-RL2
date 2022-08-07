# pylint: disable=wrong-import-order,wrong-import-position

import os
import warnings
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation

from dotenv import load_dotenv

load_dotenv()

###

# warnings.filterwarnings("ignore")

###

deprecation._PRINT_DEPRECATION_WARNINGS = False
# logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
