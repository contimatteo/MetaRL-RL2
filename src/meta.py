# pylint: disable=wrong-import-order, unused-import, consider-using-f-string
from typing import List

import utils.env_setup

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from gym.spaces import Discrete
from loguru import logger
from progress.bar import Bar
from tensorflow.python.keras.optimizers import rmsprop_v2

from agents import A3C
from agents import MetaA3C
from environments import BanditTwoArmedDependentEasy
from environments import BanditTwoArmedDependentMedium
from environments import BanditTwoArmedDependentHard
from environments import BanditTenArmedRandomRandom
from environments import BanditTenArmedRandomFixed
from networks import ActorCriticNetworks
from networks import MetaActorCriticNetworks
from policies import NetworkPolicy
from policies import RandomMetaPolicy
from policies import NetworkMetaPolicy
from utils import PlotUtils

###

RANDOM_SEED = 42

N_TRIALS_TRAIN = 25
N_TRIALS_TEST = 25
N_EPISODES_TRAIN = 1
N_EPISODES_TEST = N_EPISODES_TRAIN
N_MAX_EPISODE_STEPS = 50

TRAIN_BATCH_SIZE = 16

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

###


def __plot(agent_name, train_history, test_history):
    PlotUtils.train_test_history(
        agent_name,
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


def run(n_trials: int, n_episodes: int, envs: List[gym.Env], agent: MetaA3C, training: bool):
    ep_steps = []
    ep_actor_losses = []
    ep_critic_losses = []
    ep_rewards_tot = []
    ep_rewards_avg = []
    ep_dones_step = []

    for trial in range(n_trials):
        env = envs[trial % len(envs)]
        agent.env_sync(env)

        if training is True:
            progbar = Bar(f"[train] TRIAL {trial+1:02} -> Episodes ...", max=n_episodes)
        else:
            progbar = Bar(f" [test] TRIAL {trial+1:02} -> Episodes ...", max=n_episodes)

        #

        # if training is True:
        ### INFO: after each trial, we have to reset the RNN hidden states
        agent.reset_memory_layer_states()

        for _ in range(n_episodes):
            state = env.reset()

            if training is True:
                agent.memory.reset()

            #

            steps = 0
            done = False
            tot_reward = 0
            next_state = None
            prev_action = np.zeros(env.action_space.shape)
            prev_reward = 0.

            while not done and steps < N_MAX_EPISODE_STEPS:
                if agent.meta_algorithm:
                    trajectory = [state, prev_action, prev_reward]
                else:
                    trajectory = state

                action = agent.act(trajectory)[0]
                if isinstance(env.action_space, Discrete):
                    action = int(action)
                next_state, reward, done, _ = env.step(action)

                steps += 1
                if training is True:
                    agent.remember(steps, state, action, reward, next_state, done)

                state = next_state
                prev_action = action
                prev_reward = float(reward)
                tot_reward += reward

            actor_loss, critic_loss = 0, 0
            if training is True:
                actor_loss, critic_loss = agent.train(batch_size=TRAIN_BATCH_SIZE)

            #

            ep_steps.append(steps)
            ep_actor_losses.append(actor_loss)
            ep_critic_losses.append(critic_loss)
            ep_rewards_tot.append(tot_reward)
            # ep_rewards_avg.append(np.mean(ep_rewards_tot[-int(N_MAX_EPISODE_STEPS / 5):]))
            # ep_rewards_avg.append(np.mean(ep_rewards_tot))
            ep_rewards_avg.append(np.mean(ep_rewards_tot[-25:]))
            ep_dones_step.append(steps)

            progbar.next()

        progbar.finish()

    #

    agent.memory.reset()

    return {
        "n_episodes": n_trials * n_episodes,
        "actor_loss": ep_actor_losses,
        "critic_loss": ep_critic_losses,
        "reward_tot": ep_rewards_tot,
        "reward_avg": ep_rewards_avg,
        "dones_step": ep_dones_step,
    }


###


def main():
    envs = [
        gym.make("Ant-v4"),
        # gym.make("LunarLander-v2"),
        # BanditEnv(p_dist=[0.3, 0.7], r_dist=[0, 1]),
        # BanditEnv(p_dist=[0.9, 0.1], r_dist=[1, 0]),
        # BanditTwoArmedDependentEasy(),
        # BanditTwoArmedDependentMedium(),
        # BanditTwoArmedDependentHard(),
        # BanditTenArmedRandomRandom(),
        # BanditTwoArmedDependentEasy(),
        # BanditTwoArmedDependentMedium(),
        # BanditTwoArmedDependentHard(),
    ]

    observation_space = envs[0].observation_space
    action_space = envs[0].action_space

    action_bounds = None
    if not isinstance(envs[0].action_space, Discrete):
        action_bounds = [action_space.low, action_space.high]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # random_policy = RandomMetaPolicy(state_space=observation_space, action_space=action_space)

    # a3crandom = MetaA3C(
    #     n_max_episode_steps=N_MAX_EPISODE_STEPS,
    #     policy=random_policy,
    #     actor_network=None,
    #     critic_network=None,
    #     actor_network_opt=None,
    #     critic_network_opt=None,
    #     memory_network=None,
    # )

    # tf.keras.backend.clear_session()
    # a3crandom_train_history = run(N_TRIALS_TRAIN, N_EPISODES_TRAIN, envs, a3crandom, training=False)
    # a3crandom_test_history = run(N_TRIALS_TEST, N_EPISODES_TEST, envs, a3crandom, training=False)
    # __plot(a3crandom.name, a3crandom_train_history, a3crandom_test_history)
    # print("\n")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # a3c_actor_network, a3c_critic_network = ActorCriticNetworks(
    #     observation_space, action_space, shared_backbone=False
    # )
    # a3c_policy = NetworkPolicy(
    #     state_space=observation_space, action_space=action_space, network=a3c_actor_network
    # )
    # a3c_actor_network_opt = rmsprop_v2.RMSprop(learning_rate=1e-4)
    # a3c_critic_network_opt = rmsprop_v2.RMSprop(learning_rate=1e-4)

    # a3c = A3C(
    #     n_max_episode_steps=N_MAX_EPISODE_STEPS,
    #     policy=a3c_policy,
    #     actor_network=a3c_actor_network,
    #     critic_network=a3c_critic_network,
    #     actor_network_opt=a3c_actor_network_opt,
    #     critic_network_opt=a3c_critic_network_opt,
    #     standardize_advantage_estimate=True
    # )

    # tf.keras.backend.clear_session()
    # a3c_train_history = run(N_TRIALS_TRAIN, N_EPISODES_TRAIN, envs, a3c, training=True)
    # a3c_test_history = run(N_TRIALS_TEST, N_EPISODES_TEST, envs, a3c, training=False)
    # __plot(a3c.name, a3c_train_history, a3c_test_history)
    # print("\n")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    a3cmeta_actor_nn, a3cmeta_critic_nn, a3cmeta_memory_nn = MetaActorCriticNetworks(
        observation_space, action_space, shared_backbone=False
    )

    a3cmeta_train_policy = RandomMetaPolicy(
        state_space=observation_space, action_space=action_space, action_buonds=action_bounds
    )
    a3cmeta_test_policy = NetworkMetaPolicy(
        state_space=observation_space,
        action_space=action_space,
        network=a3cmeta_actor_nn,
        action_buonds=action_bounds
    )

    a3cmeta_actor_nn_opt = rmsprop_v2.RMSprop(learning_rate=1e-4)
    a3cmeta_critic_nn_opt = rmsprop_v2.RMSprop(learning_rate=1e-4)

    a3cmeta = MetaA3C(
        n_max_episode_steps=N_MAX_EPISODE_STEPS,
        policy=a3cmeta_train_policy,
        actor_network=a3cmeta_actor_nn,
        critic_network=a3cmeta_critic_nn,
        actor_network_opt=a3cmeta_actor_nn_opt,
        critic_network_opt=a3cmeta_critic_nn_opt,
        memory_network=a3cmeta_memory_nn,
    )

    tf.keras.backend.clear_session()
    a3cmeta_train_history = run(N_TRIALS_TRAIN, N_EPISODES_TRAIN, envs, a3cmeta, training=True)
    a3cmeta.policy = a3cmeta_test_policy
    a3cmeta_test_history = run(N_TRIALS_TEST, N_EPISODES_TEST, envs, a3cmeta, training=False)
    __plot(a3cmeta.name, a3cmeta_train_history, a3cmeta_test_history)
    print("\n")


###

if __name__ == "__main__":
    main()
