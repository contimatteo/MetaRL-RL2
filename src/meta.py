# pylint: disable=wrong-import-order, unused-import, consider-using-f-string
from typing import List

import utils.env_setup

import random
import gym
import numpy as np
import tensorflow as tf

from loguru import logger
from progress.bar import Bar

from agents import A3C, A3CMeta
from environments import BanditEnv
from networks import MetaActorCriticNetworks
from policies import RandomMetaPolicy, NetworkMetaPolicy

###

RANDOM_SEED = 666

TRAIN_BATCH_SIZE = 8

N_TRIALS = 2
N_EPISODES = 5
N_MAX_EPISODE_STEPS = 400

###


def run_agent(envs: List[gym.Env], agent: A3CMeta):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    ### TRAIN

    history = []

    print("\n")

    for trial in range(N_TRIALS):
        env = envs[trial % len(envs)]
        agent.env_sync(env)

        ep_progbar = Bar(f"TRIAL {trial} -> Episodes ...", max=N_EPISODES)

        #

        meta_memory_states = None

        ### INFO: after each trial, we have to reset the RNN hidden states
        agent.reset_memory_layer_states()

        #

        for episode in range(N_EPISODES):
            agent.memory.reset()

            ### INFO: each episode has a finite number of batches.
            ### My guess is that despite the `LSTM` `stateful` parameter, after the completion of
            ### the training on a finite number of batches (generated from a single episode) the
            ### `LSTM` does not preserve the values of its `states`.
            ### For the reason explained above, I'm going to save after each episode the values of
            ### the `LSTM.states` and then I pre-load these ones before the following episode.
            assert (episode > 0) or (episode == 0 and meta_memory_states is None)
            assert (episode < 1) or (episode > 0 and meta_memory_states is not None)
            if meta_memory_states is not None:
                ### this is not the first episode of the current trial
                agent.set_meta_memory_layer_states(meta_memory_states)

            step = 0
            done = False
            next_state = None
            state, _ = env.reset(seed=RANDOM_SEED, return_info=True)

            prev_action = 0.
            prev_reward = 0.

            while not done and step < N_MAX_EPISODE_STEPS:
                step += 1

                trajectory = [np.array(state), prev_action, prev_reward]
                action = int(agent.act(trajectory)[0])

                next_state, reward, done, _ = env.step(action)
                # logger.debug(f" > step = {step}, action = {action}, reward = {reward}, done = {done}")

                agent.remember(step, state, action, reward, next_state, done)
                state = next_state

            ### TRAIN
            episode_metrics = agent.train(batch_size=TRAIN_BATCH_SIZE)

            ### INFO: persist
            meta_memory_states = agent.get_meta_memory_layer_states()

            history.append(episode_metrics)
            ep_progbar.next()

        #

        ep_progbar.finish()

    #

    print("\n")

    for episode_metrics in history:
        Al_avg = episode_metrics["actor_nn_loss_avg"]
        # Al_sum = episode_metrics["actor_nn_loss_sum"]
        Cl_avg = episode_metrics["critic_nn_loss_avg"]
        # Cl_sum = episode_metrics["critic_nn_loss_sum"]
        R_avg = episode_metrics["rewards_avg"]
        R_sum = episode_metrics["rewards_sum"]
        logger.debug(
            "> Al_avg = {:.3f}, Cl_avg = {:.3f}, R_avg = {:.3f}, R_sum = {:.3f}".format(
                Al_avg, Cl_avg, R_avg, R_sum
            )
        )

    print("\n")


###


def main():
    ### ENV

    envs = [
        # BanditEnv(p_dist=[0.3, 0.7], r_dist=[1, 1]),
        # BanditEnv(p_dist=[0.5, 0.5], r_dist=[1, 1]),
        # BanditEnv(p_dist=[0.9, 0.1], r_dist=[1, 1]),
        gym.make("LunarLander-v2"),
    ]

    observation_space = envs[0].observation_space
    action_space = envs[0].action_space

    #

    actor_network, critic_network = MetaActorCriticNetworks(
        observation_space, action_space, TRAIN_BATCH_SIZE
    )

    policy = RandomMetaPolicy(state_space=observation_space, action_space=action_space)
    # policy = NetworkMetaPolicy(
    #     state_space=observation_space, action_space=action_space, network=actor_network
    # )

    meta = A3CMeta(
        n_max_episode_steps=N_MAX_EPISODE_STEPS,
        policy=policy,
        actor_network=actor_network,
        critic_network=critic_network,
        # opt_gradient_clip_norm=999.0  # 0.25
    )

    #

    run_agent(envs, meta)


###

if __name__ == "__main__":
    main()
