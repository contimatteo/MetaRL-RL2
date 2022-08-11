# pylint: disable=wrong-import-order, unused-import, consider-using-f-string
import utils.env_setup

import random
import gym
import numpy as np
import tensorflow as tf

from loguru import logger
from progress.bar import Bar

from agents import A2C
from networks import ActorNetwork, CriticNetwork
from policies import NetworkPolicy

###

RANDOM_SEED = 666

ENV_RENDER = False
# ENV_NAME = "MountainCar-v0"
# ENV_NAME = "CartPole-v1"
ENV_NAME = "LunarLander-v2"

N_EPISODES = 10
N_MAX_EPISODE_STEPS = 25
N_EPISODE_STEP_SECONDS_DELAY = .3

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

        state, _ = env.reset(seed=42, return_info=True)
        ENV_RENDER and env.render()

        agent.memory.reset()

        while not done and step < N_MAX_EPISODE_STEPS:
            step += 1
            action = int(agent.act(state)[0])

            next_state, reward, done, _ = env.step(action)
            ENV_RENDER and env.render()
            # logger.debug(f" > step = {step}, action = {action}, reward = {reward}, done = {done}")

            agent.remember(step, state, action, reward, next_state, done)
            state = next_state

        episode_metrics = agent.train(batch_size=8)
        history.append(episode_metrics)

        progbar.next()

    progbar.finish()

    #

    for episode_metrics in history:
        Al_avg = episode_metrics["actor_nn_loss_avg"]
        Al_sum = episode_metrics["actor_nn_loss_sum"]
        Cl_avg = episode_metrics["critic_nn_loss_avg"]
        Cl_sum = episode_metrics["critic_nn_loss_sum"]
        R_avg = episode_metrics["rewards_avg"]
        logger.debug(
            "> Al_avg = {:.3f}, Al_sum = {:.3f} Cl_avg = {:.3f}, Cl_sum = {:.3f}, R_avg = {:.3f}".
            format(Al_avg, Al_sum, Cl_avg, Cl_sum, R_avg)
        )

    print("\n")


###


def main():
    env = gym.make(ENV_NAME)

    state_space = env.observation_space
    action_space = env.action_space

    #

    actor_network = ActorNetwork(n_actions=env.action_space.n)
    critic_network = CriticNetwork()

    policy = NetworkPolicy(
        state_space=state_space, action_space=action_space, network=actor_network
    )

    new_agent = A2C(
        n_max_episode_steps=N_MAX_EPISODE_STEPS,
        policy=policy,
        actor_network=actor_network,
        critic_network=critic_network
    )

    run_agent(env, new_agent)


###

if __name__ == "__main__":
    main()
