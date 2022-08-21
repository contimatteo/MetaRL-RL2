import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

###


class PlotUtils:

    @staticmethod
    def interpolate(x, y, k):
        window_length = int(len(x) / k)
        # return make_interp_spline(x, y, k=k)(x)
        return savgol_filter(y, window_length, 3)

    @staticmethod
    def train_test_history(agent_name, history: dict):
        fig, axs = plt.subplots(2, 2)

        fig.canvas.manager.set_window_title(agent_name)

        train_ep = np.arange(0, history["train_n_episodes"], 1)
        test_ep = np.arange(0, history["test_n_episodes"], 1)

        train_actor_loss = history["train_actor_loss"]
        train_critic_loss = history["train_critic_loss"]
        train_rewards_avg = history["train_reward_avg"]
        train_dones_step = history["train_done_step"]
        test_rewards_avg = history["test_reward_avg"]
        test_dones_step = history["test_done_step"]

        train_actor_loss = PlotUtils.interpolate(train_ep, train_actor_loss, k=5)
        train_critic_loss = PlotUtils.interpolate(train_ep, train_critic_loss, k=5)
        train_rewards_avg = PlotUtils.interpolate(train_ep, train_rewards_avg, k=5)
        train_dones_step = PlotUtils.interpolate(train_ep, train_dones_step, k=5)
        test_rewards_avg = PlotUtils.interpolate(test_ep, test_rewards_avg, k=5)
        test_dones_step = PlotUtils.interpolate(test_ep, test_dones_step, k=5)

        axs[0, 0].plot(train_ep, train_actor_loss, 'tab:red')
        axs[0, 0].set_title('[train] Actor Loss (avg)')

        axs[0, 1].plot(train_ep, train_critic_loss, 'tab:red')
        axs[0, 1].set_title('[train] Critic Loss (avg)')

        axs[1, 0].set_title('Rewards (avg)')
        axs[1, 0].plot(train_ep, train_rewards_avg, label="train")
        axs[1, 0].plot(test_ep, test_rewards_avg, label="test")
        axs[1, 0].legend()

        axs[1, 1].set_title('Dones (#steps)')
        axs[1, 1].plot(train_ep, train_dones_step, label="train")
        axs[1, 1].plot(test_ep, test_dones_step, label="test")
        axs[1, 1].legend()
