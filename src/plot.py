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
    if len(configs) == 8:
        fig, axs = plt.subplots(2, 4)
        xlabel_rows_idx = [4, 5, 6, 7]
        ylabel_rows_idx = [0, 4]
    else:
        fig, axs = plt.subplots(2, 3)
        xlabel_rows_idx = [3, 4, 5]
        ylabel_rows_idx = [0, 3]

    # fig.tight_layout()
    # fig.set_size_inches(10, 5.5)
    fig.set_dpi(200)
    # plt.subplots_adjust(wspace=0.3, hspace=0.2)

    for i, (ax, config) in enumerate(zip(axs.flat, configs)):
        title = config["title"]
        ylabel = config["ylabel"]
        trials = config["trials"]

        ax.set_title(title)
        ax.set(xlabel='Episodes', ylabel=ylabel)
        # Hide x/y labels and tick labels
        if i not in xlabel_rows_idx:
            ax.xaxis.label.set_visible(False)
        if i not in ylabel_rows_idx:
            ax.yaxis.label.set_visible(False)

        for trial in trials:
            y = trial["data"]
            x = np.arange(0, trial["n_episodes"], 1)
            y = PlotUtils.interpolate(x, y, k=10)
            ax.plot(x, y, label=trial["label"], linewidth=1.5)

    handles, labels = axs.flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, labelspacing=0.)

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

    assert isinstance(config, list) or isinstance(config, dict)

    if isinstance(config, dict):
        config = __load_trials(config)
        __plots(config)
    elif isinstance(config, list):
        assert len(config) == 8 or len(config) == 6
        config = __load_multi_trials(config)
        __advanced_plot(config)
    else:
        raise Exception("config invalid.")


###

if __name__ == '__main__':
    main(parse_args())
