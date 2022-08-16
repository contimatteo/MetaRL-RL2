import utils.env_setup

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
