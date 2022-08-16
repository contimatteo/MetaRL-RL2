# pylint: disable=wrong-import-order, unused-import, consider-using-f-string
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from gym.spaces import Discrete
from progress.bar import Bar

from utils import PlotUtils

from .controller import Controller

###


class StandardController(Controller):

    def __trajectory(self, state, prev_action, prev_reward) -> Any:
        if self.agent.meta_algorithm:
            trajectory = [state, prev_action, prev_reward]
        else:
            trajectory = state

        return trajectory

    def __action(self, trajectory) -> Any:
        action = self.agent.act(trajectory)[0]

        if self.env_actions_are_discrete:
            action = int(action)

        return action

    #

    def __train(self) -> dict:
        ep_steps = []
        ep_actor_losses = []
        ep_critic_losses = []
        ep_rewards_tot = []
        ep_rewards_avg = []
        ep_dones_step = []

        n_trials = self._config.n_trials
        n_episodes = self._config.n_episodes

        ### EXPLORATION

        n_exploration_episodes = self._config.n_explore_episodes

        for trial in range(n_trials):
            env = self.envs[trial % len(self.envs)]
            self.agent.env_sync(env)

            progbar = Bar(
                f"[explore] TRIAL {trial+1:02} -> Episodes ...", max=n_exploration_episodes
            )

            for _ in range(n_exploration_episodes):
                state = env.reset()
                self.agent.memory.reset()

                steps = 0
                done = False
                next_state = None

                while not done and steps < self._config.n_max_steps:
                    action = env.action_space.sample()
                    next_state, reward, done, _ = env.step(action)
                    steps += 1
                    self.agent.remember(steps, state, action, reward, next_state, done)
                    state = next_state

                _, _ = self.agent.train(batch_size=self._config.batch_size)

                progbar.next()
            progbar.finish()

        ###

        for trial in range(n_trials):
            env = self.envs[trial % len(self.envs)]
            self.agent.env_sync(env)

            progbar = Bar(f"[train] TRIAL {trial+1:02} -> Episodes ...", max=n_episodes)

            ### INFO: after each trial, we have to reset the RNN hidden states
            self.agent.reset_memory_layer_states()

            for _ in range(n_episodes):
                state = env.reset()
                self.agent.memory.reset()

                steps = 0
                done = False
                tot_reward = 0
                next_state = None
                prev_action = np.zeros(env.action_space.shape)
                prev_reward = 0.

                while not done and steps < self._config.n_max_steps:
                    trajectory = self.__trajectory(state, prev_action, prev_reward)
                    action = self.__action(trajectory)
                    next_state, reward, done, _ = env.step(action)

                    steps += 1
                    self.agent.remember(steps, state, action, reward, next_state, done)

                    state = next_state
                    prev_action = action
                    prev_reward = float(reward)
                    tot_reward += reward

                actor_loss, critic_loss = self.agent.train(batch_size=self._config.batch_size)

                ep_steps.append(steps)
                ep_actor_losses.append(actor_loss)
                ep_critic_losses.append(critic_loss)
                ep_rewards_tot.append(tot_reward)
                ep_rewards_avg.append(np.mean(ep_rewards_tot[-25:]))
                ep_dones_step.append(steps)

                progbar.next()
            progbar.finish()

        #

        self.agent.memory.reset()

        return {
            "n_episodes": n_trials * n_episodes,
            "actor_loss": ep_actor_losses,
            "critic_loss": ep_critic_losses,
            "reward_tot": ep_rewards_tot,
            "reward_avg": ep_rewards_avg,
            "dones_step": ep_dones_step,
        }

    def __inference(self) -> dict:
        ep_steps = []
        ep_actor_losses = []
        ep_critic_losses = []
        ep_rewards_tot = []
        ep_rewards_avg = []
        ep_dones_step = []

        n_trials = self._config.n_trials
        n_episodes = self._config.n_episodes

        #

        for trial in range(n_trials):
            env = self.envs[trial % len(self.envs)]
            self.agent.env_sync(env)

            progbar = Bar(f" [test] TRIAL {trial+1:02} -> Episodes ...", max=n_episodes)

            ### INFO: after each trial, we have to reset the RNN hidden states
            self.agent.reset_memory_layer_states()

            for _ in range(n_episodes):
                state = env.reset()

                steps = 0
                done = False
                tot_reward = 0
                next_state = None
                prev_action = np.zeros(env.action_space.shape)
                prev_reward = 0.

                while not done and steps < self._config.n_max_steps:
                    trajectory = self.__trajectory(state, prev_action, prev_reward)
                    action = self.__action(trajectory)
                    next_state, reward, done, _ = env.step(action)

                    steps += 1

                    state = next_state
                    prev_action = action
                    prev_reward = float(reward)
                    tot_reward += reward

                ep_steps.append(steps)
                ep_actor_losses.append(0.)
                ep_critic_losses.append(0.)
                ep_rewards_tot.append(tot_reward)
                ep_rewards_avg.append(np.mean(ep_rewards_tot[-25:]))
                ep_dones_step.append(steps)

                progbar.next()
            progbar.finish()

        #

        self.agent.memory.reset()

        return {
            "n_episodes": n_trials * n_episodes,
            "actor_loss": ep_actor_losses,
            "critic_loss": ep_critic_losses,
            "reward_tot": ep_rewards_tot,
            "reward_avg": ep_rewards_avg,
            "dones_step": ep_dones_step,
        }

    def __render(self) -> dict:
        return {}

    #

    def __plot(self, train_history, test_history):
        PlotUtils.train_test_history(
            f"{self.agent.name} ({self._config.policy})",
            {
                ### train
                "train_n_episodes": train_history["n_episodes"],
                "train_actor_loss": train_history["actor_loss"],
                "train_critic_loss": train_history["critic_loss"],
                "train_reward_sum": train_history["reward_tot"],
                "train_reward_avg": train_history["reward_avg"],
                "train_done_step": train_history["dones_step"],
                ### test
                "test_n_episodes": test_history["n_episodes"],
                "test_actor_loss": test_history["actor_loss"],
                "test_critic_loss": test_history["critic_loss"],
                "test_reward_avg": test_history["reward_avg"],
                "test_done_step": test_history["dones_step"],
            }
        )

        plt.show()

    def __save_history(self, history) -> None:
        pass

    #

    def run(self) -> None:
        super().run()

        if self.mode == "training":
            history = self.__train()

            test_history = self.__inference()
            self.__plot(history, test_history)

            self.__save_history(history)

        elif self.mode == "inference":
            history = self.__inference()

        elif self.mode == "render":
            history = self.__render()

        else:
            raise Exception("mode not supported.")

        self.__save_history(history)
