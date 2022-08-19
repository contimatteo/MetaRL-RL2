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

    def __load_trial_json(trial: dict) -> dict:
        _url = LocalStorageManager.dirs.tmp_history
        _url = _url.joinpath(trial["mode"])
        _url = _url.joinpath(trial["id"] + ".json")

        assert _url.exists()
        assert _url.is_file()

        with open(str(_url), 'r', encoding="utf-8") as file:
            return json.load(file)

    for i, trial in enumerate(configs["trials"]):
        data = __load_trial_json(trial)
        configs["trials"][i]["n_episodes"] = data["n_episodes"]
        configs["trials"][i]["data"] = data[trial["metric"]]

    return configs


def __load_multi_trials(configs: list) -> List[list]:

    def __load_trial_json(trial: dict) -> dict:
        _url = LocalStorageManager.dirs.tmp_history
        _url = _url.joinpath(trial["mode"])
        _url = _url.joinpath(trial["id"] + ".json")

        assert _url.exists()
        assert _url.is_file()

        with open(str(_url), 'r', encoding="utf-8") as file:
            return json.load(file)

    for ic, config in enumerate(configs):
        for it, trial in enumerate(config["trials"]):
            data = __load_trial_json(trial)
            configs[ic]["trials"][it]["n_episodes"] = data["n_episodes"]
            configs[ic]["trials"][it]["data"] = data[trial["metric"]]

    return configs


def __plots(configs) -> None:
    title = configs["title"]
    ylabel = configs["ylabel"]
    trials = configs["trials"]

    fig = plt.figure()

    plt.title(title)
    fig.canvas.manager.set_window_title(title)

    plt.xlabel("Episodes")
    plt.ylabel(ylabel)

    #

    for trial in trials:
        y = trial["data"]
        x = np.arange(0, trial["n_episodes"], 1)
        # y = PlotUtils.interpolate(x, y, k=3)
        plt.plot(x, y, label=trial["label"], linewidth=2.0)

    # plt.legend()
    plt.legend(fontsize=12)  # using a size in points
    plt.show()


def __advanced_plot(configs: List[dict]) -> None:
    fig, axs = plt.subplots(2, 3)
    # plt.title(title)
    # fig.canvas.manager.set_window_title(title)

    for ax, config in zip(axs.flat, configs):
        title = config["title"]
        ylabel = config["ylabel"]
        trials = config["trials"]

        ax.set_title(title)
        ax.set(xlabel='Episodes', ylabel=ylabel)
        ax.label_outer()  # Hide x/y labels and tick labels

        for trial in trials:
            y = trial["data"]
            x = np.arange(0, trial["n_episodes"], 1)
            # y = PlotUtils.interpolate(x, y, k=3)
            ax.plot(x, y, label=trial["label"], linewidth=2.0)

    # plt.legend()
    # plt.legend(fontsize=12)  # using a size in points
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

    assert isinstance(config, list)
    assert len(config) == 6

    #

    config = __load_multi_trials(config)

    __advanced_plot(config)


###

if __name__ == '__main__':
    main(parse_args())
