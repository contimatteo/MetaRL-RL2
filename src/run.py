import utils.env_setup

# import numpy as np
# import tensorflow as tf
# RANDOM_SEED = 42
# np.random.seed(RANDOM_SEED)
# tf.random.set_seed(RANDOM_SEED)

from core import StandardController
from utils import parse_args

###


def main(args):
    controller = StandardController()

    controller.intialize(args.config)

    controller.run()


###

if __name__ == '__main__':
    main(parse_args())
