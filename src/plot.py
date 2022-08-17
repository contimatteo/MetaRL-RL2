from typing import List

import pathlib
import json
import matplotlib.pyplot as plt
import numpy as np

from utils import parse_args
from utils import LocalStorageManager
from utils import PlotUtils

###


def __load_trials(configs: dict) -> List[dict]:

    def __load_trial_json(url: str) -> dict:
        _url = LocalStorageManager.dirs.configs.joinpath(url)
        assert _url.exists()
        assert _url.is_file()
        with open(str(_url), 'r', encoding="utf-8") as file:
            return json.load(file)

    trials = []

    for trial_config in configs["trials"]:
        trial_data = __load_trial_json(trial_config["file_url"])
        trials.append(trial_data)

    return trials


def __plots(configs, trials_data: List[dict]) -> None:
    fig = plt.figure()

    fig.canvas.manager.set_window_title(configs["title"])

    for _ in trials_data:
        y = [10, 20, 30, 20, 10]
        x = np.arange(0, len(y), 1)
        y = PlotUtils.interpolate(x, y, k=1)
        plt.plot(x, y, label="train")

    plt.legend()
    plt.show()


###


def main(args):
    config_file_url = args.config
    assert isinstance(config_file_url, pathlib.Path)

    assert config_file_url.exists()
    assert config_file_url.is_file()

    config = None
    with open(str(config_file_url), 'r', encoding="utf-8") as file:
        config = json.load(file)
    assert isinstance(config, dict)

    #

    trials_data = __load_trials(config)

    __plots(config, [None])


###

if __name__ == '__main__':
    main(parse_args())
