# pylint: disable=wrong-import-order, unused-import, consider-using-f-string
import utils.env_setup

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

from loguru import logger
from progress.bar import Bar
from tensorflow.python.keras.optimizers import rmsprop_v2

from agents import AC
from agents import A2C
from agents import A3C
from agents import MetaA3C
from networks import ActorCriticNetworks
from networks import MetaActorCriticNetworks
from policies import NetworkPolicy
from policies import NetworkMetaPolicy
from utils import PlotUtils

###

RANDOM_SEED = 42

ENV_NAME = "CartPole-v0"
# ENV_NAME = "LunarLander-v2"

N_EPISODES_TRAIN = 25
N_EPISODES_TEST = N_EPISODES_TRAIN
N_MAX_EPISODE_STEPS = 10000

TRAIN_BATCH_SIZE = 32

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
            ### test
            "test_n_episodes": test_history["n_episodes"],
            "test_actor_loss": test_history["actor_loss"],
            "test_critic_loss": test_history["critic_loss"],
            "test_reward_sum": test_history["reward_tot"],
            "test_reward_avg": test_history["reward_avg"],
        }
    )

    plt.show()


def run(n_episodes: int, env: gym.Env, agent: A2C, training: bool):
    ep_steps = []
    ep_actor_losses = []
    ep_critic_losses = []
    ep_rewards_tot = []
    ep_rewards_avg = []

    if training is True:
        progbar = Bar('[train] Episodes ...', max=n_episodes)
    else:
        progbar = Bar(' [test] Episodes ...', max=n_episodes)

    for _ in range(n_episodes):
        state = env.reset()

        if training is True:
            agent.memory.reset()

        #

        steps = 0
        done = False
        tot_reward = 0
        next_state = None
        prev_action = 0
        prev_reward = 0.

        while not done and steps < N_MAX_EPISODE_STEPS:
            if agent.meta_algorithm:
                trajectory = [state, prev_action, prev_reward]
            else:
                trajectory = np.array([state])

            action = int(agent.act(trajectory)[0])
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
        if training is True:
            ep_rewards_avg.append(np.mean(ep_rewards_tot[-100:]))
        else:
            ep_rewards_avg.append(np.mean(ep_rewards_tot))

        progbar.next()

    #

    progbar.finish()

    agent.memory.reset()

    return {
        "n_episodes": n_episodes,
        "actor_loss": ep_actor_losses,
        "critic_loss": ep_critic_losses,
        "reward_tot": ep_rewards_tot,
        "reward_avg": ep_rewards_avg,
    }


###


def main():
    env = gym.make(ENV_NAME)

    observation_space = env.observation_space
    action_space = env.action_space

    ###

    # a2c_actor_network, a2c_critic_network = ActorCriticNetworks(
    #     observation_space, action_space, shared_backbone=False
    # )
    # a2c_policy = NetworkPolicy(
    #     state_space=observation_space, action_space=action_space, network=a2c_actor_network
    # )
    # a2c_actor_network_opt = rmsprop_v2.RMSprop(learning_rate=1e-4)
    # a2c_critic_network_opt = rmsprop_v2.RMSprop(learning_rate=1e-4)
    # a2c = A2C(
    #     n_max_episode_steps=N_MAX_EPISODE_STEPS,
    #     policy=a2c_policy,
    #     actor_network=a2c_actor_network,
    #     critic_network=a2c_critic_network,
    #     actor_network_opt=a2c_actor_network_opt,
    #     critic_network_opt=a2c_critic_network_opt,
    #     standardize_advantage_estimate=False
    # )
    # tf.keras.backend.clear_session()
    # a2c_train_history = run(N_EPISODES_TRAIN, env, a2c, training=True)
    # a2c_test_history = run(N_EPISODES_TEST, env, a2c, training=False)
    # __plot(a2c.name, a2c_train_history, a2c_test_history)
    # print("\n")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    a3c_actor_network, a3c_critic_network = ActorCriticNetworks(
        observation_space, action_space, shared_backbone=False
    )

    a3c_policy = NetworkPolicy(
        state_space=observation_space, action_space=action_space, network=a3c_actor_network
    )

    a3c_actor_network_opt = rmsprop_v2.RMSprop(learning_rate=1e-4)
    a3c_critic_network_opt = rmsprop_v2.RMSprop(learning_rate=1e-4)

    a3c = A3C(
        n_max_episode_steps=N_MAX_EPISODE_STEPS,
        policy=a3c_policy,
        actor_network=a3c_actor_network,
        critic_network=a3c_critic_network,
        actor_network_opt=a3c_actor_network_opt,
        critic_network_opt=a3c_critic_network_opt,
        standardize_advantage_estimate=True
    )

    tf.keras.backend.clear_session()
    a3c_train_history = run(N_EPISODES_TRAIN, env, a3c, training=True)
    a3c_test_history = run(N_EPISODES_TEST, env, a3c, training=False)
    __plot(a3c.name, a3c_train_history, a3c_test_history)
    print("\n")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # a3cmeta_actor_nn, a3cmeta_critic_nn, a3cmeta_memory_nn = MetaActorCriticNetworks(
    #     observation_space, action_space, batch_size=TRAIN_BATCH_SIZE
    # )

    # a3cmeta_policy = NetworkMetaPolicy(
    #     state_space=observation_space, action_space=action_space, network=a3cmeta_actor_nn
    # )

    # a3cmeta_actor_nn_opt = rmsprop_v2.RMSprop(learning_rate=1e-4)
    # a3cmeta_critic_nn_opt = rmsprop_v2.RMSprop(learning_rate=1e-4)

    # a3cmeta = MetaA3C(
    #     n_max_episode_steps=N_MAX_EPISODE_STEPS,
    #     policy=a3cmeta_policy,
    #     actor_network=a3cmeta_actor_nn,
    #     critic_network=a3cmeta_critic_nn,
    #     actor_network_opt=a3cmeta_actor_nn_opt,
    #     critic_network_opt=a3cmeta_critic_nn_opt,
    #     memory_network=a3cmeta_memory_nn,
    #     standardize_advantage_estimate=True
    # )

    # tf.keras.backend.clear_session()
    # a3cmeta_train_history = run(N_EPISODES_TRAIN, env, a3cmeta, training=True)
    # a3cmeta_test_history = run(N_EPISODES_TEST, env, a3cmeta, training=False)
    # __plot(a3cmeta.name, a3cmeta_train_history, a3cmeta_test_history)
    # print("\n")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


###

if __name__ == "__main__":
    main()
