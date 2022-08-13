# pylint: disable=wrong-import-order, unused-import, consider-using-f-string
import utils.env_setup

import random
import gym
import numpy as np
import tensorflow as tf

from loguru import logger
from progress.bar import Bar

from agents import A2C, A3C
from networks import ActorCriticNetworks
from policies import NetworkPolicy

###

RANDOM_SEED = 666

# ENV_NAME = "MountainCar-v0"
# ENV_NAME = "CartPole-v1"
ENV_NAME = "LunarLander-v2"

N_EPISODES = 10
N_MAX_EPISODE_STEPS = 400

TRAIN_BATCH_SIZE = 8

###


def run_agent(env, agent: A2C):
    env.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    ### TRAIN

    history = []

    print("\n")
    progbar = Bar('Running Episodes ...', max=N_EPISODES)

    for _ in range(N_EPISODES):
        step = 0
        done = False
        next_state = None

        state, _ = env.reset(seed=RANDOM_SEED, return_info=True)

        agent.memory.reset()

        while not done and step < N_MAX_EPISODE_STEPS:
            step += 1
            action = int(agent.act(state)[0])

            next_state, reward, done, _ = env.step(action)
            # logger.debug(f" > step = {step}, action = {action}, reward = {reward}, done = {done}")

            agent.remember(step, state, action, reward, next_state, done)
            state = next_state

        episode_metrics = agent.train(batch_size=TRAIN_BATCH_SIZE)
        history.append(episode_metrics)

        progbar.next()

    progbar.finish()

    #

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
    env = gym.make(ENV_NAME)

    observation_space = env.observation_space
    action_space = env.action_space

    ###

    a2c_actor_network, a2c_critic_network = ActorCriticNetworks(observation_space, action_space)
    a2c_policy = NetworkPolicy(
        state_space=observation_space, action_space=action_space, network=a2c_actor_network
    )
    a2c = A2C(
        n_max_episode_steps=N_MAX_EPISODE_STEPS,
        policy=a2c_policy,
        actor_network=a2c_actor_network,
        critic_network=a2c_critic_network,
        opt_gradient_clip_norm=999.0  # 0.25
    )

    #

    a3c_actor_network, a3c_critic_network = ActorCriticNetworks(observation_space, action_space)
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

    # run_agent(env, a2c)
    run_agent(env, a3c)


###

if __name__ == "__main__":
    main()
