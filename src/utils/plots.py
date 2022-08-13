import math
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import make_interp_spline

###


class PlotUtils:

    @staticmethod
    def model_training_overview(eps_history: list):
        fig, axs = plt.subplots(2, 2)

        n_episodes = len(eps_history)

        episodes = np.arange(0, n_episodes, 1)
        actor_loss_avg = []
        critic_loss_avg = []
        rewards_avg = []
        rewards_sum = []

        for episode_metrics in eps_history:
            actor_loss_avg.append(episode_metrics["actor_nn_loss_avg"])
            critic_loss_avg.append(episode_metrics["critic_nn_loss_avg"])
            rewards_avg.append(episode_metrics["rewards_avg"])
            rewards_sum.append(episode_metrics["rewards_sum"])

        actor_loss_avg = make_interp_spline(episodes, actor_loss_avg, k=3)(episodes)
        critic_loss_avg = make_interp_spline(episodes, critic_loss_avg, k=3)(episodes)
        rewards_avg = make_interp_spline(episodes, rewards_avg, k=3)(episodes)
        rewards_sum = make_interp_spline(episodes, rewards_sum, k=3)(episodes)

        axs[0, 0].plot(episodes, actor_loss_avg)
        axs[0, 0].set_title('[episode] Actor Loss (avg)')

        axs[0, 1].plot(episodes, critic_loss_avg, 'tab:orange')
        axs[0, 1].set_title('[episode] Critic Loss (avg)')

        axs[1, 0].plot(episodes, rewards_avg, 'tab:red')
        axs[1, 0].set_title('[episode] Rewards (avg)')

        axs[1, 1].plot(episodes, rewards_sum, 'tab:red')
        axs[1, 1].set_title('[episode] Rewards (sum)')

        # for ax in axs.flat:
        #     ax.set(xlabel='x-label', ylabel='y-label')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        # for ax in axs.flat:
        #     ax.label_outer()
