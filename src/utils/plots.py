import math
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import make_interp_spline

###


class PlotUtils:

    @staticmethod
    def __interpolate(x, y, k=3):
        return make_interp_spline(x, y, k=k)(x)

    @staticmethod
    def train_test_history(history: dict):
        fig, axs = plt.subplots(2, 2)

        train_ep = np.arange(0, history["train_n_episodes"], 1)
        test_ep = np.arange(0, history["test_n_episodes"], 1)

        train_actor_loss = history["train_actor_loss"]
        train_critic_loss = history["train_critic_loss"]
        train_rewards_avg = history["train_reward_avg"]
        train_rewards_sum = history["train_reward_sum"]
        test_rewards_avg = history["test_reward_avg"]
        test_rewards_sum = history["test_reward_sum"]

        train_actor_loss = PlotUtils.__interpolate(train_ep, train_actor_loss, k=3)
        train_critic_loss = PlotUtils.__interpolate(train_ep, train_critic_loss, k=3)
        train_rewards_avg = PlotUtils.__interpolate(train_ep, train_rewards_avg, k=3)
        train_rewards_sum = PlotUtils.__interpolate(train_ep, train_rewards_sum, k=3)
        test_rewards_avg = PlotUtils.__interpolate(test_ep, test_rewards_avg, k=3)
        test_rewards_sum = PlotUtils.__interpolate(test_ep, test_rewards_sum, k=3)

        axs[0, 0].plot(train_ep, train_actor_loss, 'tab:red')
        axs[0, 0].set_title('[train] Actor Loss (avg)')
        axs[0, 1].plot(train_ep, train_critic_loss, 'tab:red')
        axs[0, 1].set_title('[train] Critic Loss (avg)')

        axs[1, 0].set_title('Rewards (avg)')
        axs[1, 0].plot(train_ep, train_rewards_avg, label="train")
        axs[1, 0].plot(test_ep, test_rewards_avg, label="test")

        axs[1, 1].set_title('Rewards (sum)')
        axs[1, 1].plot(train_ep, train_rewards_sum, label="train")
        axs[1, 1].plot(test_ep, test_rewards_sum, label="test")

        # for ax in axs.flat:
        #     ax.set(xlabel='x-label', ylabel='y-label')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        # for ax in axs.flat:
        #     ax.label_outer()
