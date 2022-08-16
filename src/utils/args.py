import argparse
import pathlib

###


def parse_args():
    parser = argparse.ArgumentParser(description='MetaRL')

    parser.add_argument('--config', required=True, type=pathlib.Path, help='config file url')

    #

    args = parser.parse_args()

    #

    config_file = pathlib.Path(args.config)

    assert config_file.exists() and config_file.is_file()

    #

    return args
