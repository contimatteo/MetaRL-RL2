# pylint: disable=wrong-import-order, unused-import, consider-using-f-string
from typing import Union

import utils.env_setup

import gym
import numpy as np
import tensorflow as tf

from loguru import logger
from progress.bar import Bar, IncrementalBar

from agents import MetaAgent
from environments import BanditEnv

###

RANDOM_SEED = 666

N_TRIALS = 5
N_EPISODES = 5
N_MAX_EPISODE_STEPS = 100

###


def __seed():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)


# def __build_env(num_experiments: int, num_bandits: int):
#     means = np.random.normal(size=(num_experiments, num_bandits))
#     stdev = np.ones((num_experiments, num_bandits))
#     return TwoArmedBanditsEnv(mean=means, stddev=stdev)

###


def main():
    __seed()

    ### AGENT

    agent = MetaAgent(n_max_episode_steps=N_MAX_EPISODE_STEPS)

    ### ENV

    envs = [
        BanditEnv(p_dist=[0.3, 0.7], r_dist=[1, 1]),
        BanditEnv(p_dist=[0.5, 0.5], r_dist=[1, 1]),
        BanditEnv(p_dist=[0.9, 0.1], r_dist=[1, 1]),
    ]

    ### TRAIN

    history = []

    print("\n")

    for trial in range(N_TRIALS):
        env = envs[trial % len(envs)]
        agent.sync_env(env)

        ep_progbar = Bar(f"TRIAL {trial} -> Episodes ...", max=N_EPISODES)

        for _ in range(N_EPISODES):
            agent.reset_memory()

            step = 0
            done = False
            state, _ = env.reset(seed=RANDOM_SEED)

            while not done and step < N_MAX_EPISODE_STEPS:
                step += 1
                action = agent.act(state)

                next_state, reward, done, _ = env.step(action)
                # logger.debug(f" > step = {step}, action = {action}, reward = {reward}, done = {done}")

                agent.remember(step, state, action, reward, next_state, done)
                state = next_state

            episode_metrics = agent.train()
            history.append(episode_metrics)

            ep_progbar.next()

        ep_progbar.finish()

    #

    # for episode_metrics in history:
    #     act_loss = episode_metrics["actor_nn_loss_avg"]
    #     crt_loss = episode_metrics["critic_nn_loss_avg"]
    #     rewards_sum = episode_metrics["rewards_sum"]
    #     rewards_avg = episode_metrics["rewards_avg"]
    #     logger.debug(
    #         "A_loss = {:.3f}, C_loss = {:.3f}, rwd_sum = {:.3f}, rwd_avg = {:.3f}".format(
    #             act_loss, crt_loss, rewards_sum, rewards_avg
    #         )
    #     )

    print("\n")


###

if __name__ == "__main__":
    main()
