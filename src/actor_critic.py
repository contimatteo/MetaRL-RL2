# pylint: disable=wrong-import-order, unused-import, consider-using-f-string
import utils.env_setup

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

from loguru import logger
from progress.bar import Bar

from agents import A2C
from agents import A3C
from networks import ActorCriticNetworks
from policies import NetworkPolicy
from utils import PlotUtils

###

RANDOM_SEED = 42

ENV_NAME = "CartPole-v0"
# ENV_NAME = "MountainCar-v0"
# ENV_NAME = "LunarLander-v2"

N_EPISODES_TRAIN = 5
N_EPISODES_TEST = 5

N_MAX_EPISODE_STEPS = 10000

TRAIN_BATCH_SIZE = 32

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

###


def __plot(train_history, test_history):
    PlotUtils.train_test_history(
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

    if training:
        progbar = Bar('[train] Episodes ...', max=n_episodes)
    else:
        progbar = Bar('[test] Episodes ...', max=n_episodes)

    for _ in range(n_episodes):
        steps = 0
        tot_reward = 0

        state = env.reset()
        agent.memory.reset()

        #

        done = False
        next_state = None

        while not done and steps < N_MAX_EPISODE_STEPS:
            action = int(agent.act(state)[0])
            next_state, reward, done, _ = env.step(action)
            # logger.debug(f" > steps = {steps}, action = {action}, reward = {reward}, done = {done}")

            steps += 1
            tot_reward += reward

            agent.remember(steps, state, action, reward, next_state, done)
            state = next_state

        if training:
            actor_loss, critic_loss = agent.train(batch_size=TRAIN_BATCH_SIZE)
        else:
            actor_loss, critic_loss = 0, 0

        #

        ep_steps.append(steps)
        ep_actor_losses.append(actor_loss)
        ep_critic_losses.append(critic_loss)
        ep_rewards_tot.append(tot_reward)
        ep_rewards_avg.append(np.mean(ep_rewards_tot[-100:]))

        progbar.next()

    progbar.finish()

    #

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

    a2c_actor_network, a2c_critic_network = ActorCriticNetworks(
        observation_space, action_space, shared_backbone=False
    )
    a2c_policy = NetworkPolicy(
        state_space=observation_space, action_space=action_space, network=a2c_actor_network
    )
    a2c = A2C(
        n_max_episode_steps=N_MAX_EPISODE_STEPS,
        policy=a2c_policy,
        actor_network=a2c_actor_network,
        critic_network=a2c_critic_network,
        opt_gradient_clip_norm=999.0,
    )

    #

    a3c_actor_network, a3c_critic_network = ActorCriticNetworks(
        observation_space, action_space, shared_backbone=False
    )
    a3c_policy = NetworkPolicy(
        state_space=observation_space, action_space=action_space, network=a3c_actor_network
    )
    a3c = A3C(
        n_max_episode_steps=N_MAX_EPISODE_STEPS,
        policy=a3c_policy,
        actor_network=a3c_actor_network,
        critic_network=a3c_critic_network,
        opt_gradient_clip_norm=999.0  # 0.25
    )

    #

    tf.keras.backend.clear_session()

    a2c_train_history = run(N_EPISODES_TRAIN, env, a2c, training=True)
    a2c_test_history = run(N_EPISODES_TEST, env, a2c, training=False)

    __plot(a2c_train_history, a2c_test_history)


###

if __name__ == "__main__":
    main()
