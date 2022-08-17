from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import time

from progress.bar import Bar

from utils import PlotUtils

from .controller import Controller

###


class StandardController(Controller):

    def __trajectory(self, state, prev_action, prev_reward) -> Any:
        if self.meta_policy:
            trajectory = [state, prev_action, prev_reward]
        else:
            trajectory = state

        return trajectory

    def __action(self, trajectory) -> Any:
        return self.policy.act(trajectory)

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
        n_explore_episodes = self._config.n_explore_episodes

        #

        for trial in range(n_trials):
            env = self.envs[trial % len(self.envs)]
            self.agent.env_sync(env)

            ### INFO: after each trial, we have to reset the RNN hidden states.
            self.agent.reset_memory_layer_states()

            ### EXPLORATION

            progbar = Bar(f"[explore] TRIAL {trial+1:02} -> Episodes ...", max=n_explore_episodes)

            for _ in range(n_explore_episodes):
                state = env.reset()
                self.agent.memory.reset()

                steps, done, next_state = 0, False, None

                while not done and steps < self._config.n_max_steps:
                    action = env.action_space.sample()
                    next_state, reward, done, _ = env.step(action)
                    steps += 1
                    self.agent.remember(steps, state, action, reward, next_state, done)
                    state = next_state

                _, _ = self.agent.train(batch_size=self._config.batch_size)

                progbar.next()
            progbar.finish()

            ### TRAINING

            progbar = Bar(f"[train] TRIAL {trial+1:02} -> Episodes ...", max=n_episodes)

            for _ in range(n_episodes):
                state = env.reset()
                self.agent.memory.reset()

                steps, done, next_state = 0, False, None
                tot_reward = 0
                prev_reward = 0.
                prev_action = np.zeros(env.action_space.shape)

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

            progbar = Bar(f"[test] TRIAL {trial+1:02} -> Episodes ...", max=n_episodes)

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

        return {
            "n_episodes": n_trials * n_episodes,
            "actor_loss": ep_actor_losses,
            "critic_loss": ep_critic_losses,
            "reward_tot": ep_rewards_tot,
            "reward_avg": ep_rewards_avg,
            "dones_step": ep_dones_step,
        }

    def __render(self) -> None:
        n_trials = self._config.n_trials
        n_episodes = self._config.n_episodes

        for trial in range(n_trials):
            env = None
            env = self.envs[trial % len(self.envs)]

            for _ in range(n_episodes):
                state = env.reset()
                env.render()

                steps, done, next_state = 0, False, None
                prev_reward = 0.
                prev_action = np.zeros(env.action_space.shape)

                while not done and steps < self._config.n_max_steps:
                    env.render()
                    trajectory = self.__trajectory(state, prev_action, prev_reward)
                    action = self.__action(trajectory)
                    next_state, _, done, _ = env.step(action)
                    env.render()

                    steps += 1
                    state = next_state

            time.sleep(2)

    #

    def __plot(self, train_history, test_history):
        PlotUtils.train_test_history(
            self._config.trial_id,
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
        self._load_envs()

        if self.mode == "training":
            self._load_networks()
            self._load_policy()
            self._load_agent()
        else:
            self._load_trained_models()
            self._load_policy()

        self._validate_controller()

        #

        if self.mode == "training":
            history = self.__train()
            self._save_trained_models()
            self.__plot(history, history)

        elif self.mode == "inference":
            history = self.__inference()
            self.__plot(history, history)

        elif self.mode == "render":
            self.__render()

        else:
            raise Exception("mode not supported.")

        #

        if self.mode == "training" or self.mode == "inference":
            self.__save_history(history)
