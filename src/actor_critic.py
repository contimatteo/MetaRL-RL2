# pylint: disable=wrong-import-order, unused-import, consider-using-f-string
from typing import Union

import utils.env_setup

import gym
import numpy as np
import tensorflow as tf

from loguru import logger
from progress.bar import Bar

from agents import AdvantageActorCritic, A2C
from networks import ActorNetwork, CriticNetwork
from policies import NetworkPolicy

###

ENV_RENDER = False
# ENV_NAME = "MountainCar-v0"
# ENV_NAME = "CartPole-v1"
ENV_NAME = "LunarLander-v2"

N_EPISODES = 10
N_MAX_EPISODE_STEPS = 1000
N_EPISODE_STEP_SECONDS_DELAY = .3

###


def __seed(env: gym.Env):
    SEED = 666
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    env.seed(SEED)


###


def run_agent(env, agent):
    __seed(env)

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

        while not done:
            step += 1
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            ENV_RENDER and env.render()
            # logger.debug(f" > step = {step}, action = {action}, reward = {reward}, done = {done}")

            agent.remember(step, state, action, reward, next_state, done)
            state = next_state

        # episode_metrics, _ = agent.train()
        # history.append(episode_metrics)

        progbar.next()

    progbar.finish()

    #

    for episode_metrics in history:
        act_loss = episode_metrics["actor_nn_loss_avg"]
        crt_loss = episode_metrics["critic_nn_loss_avg"]
        rewards_sum = episode_metrics["rewards_sum"]
        rewards_avg = episode_metrics["rewards_avg"]
        logger.debug(
            "A_loss = {:.3f}, C_loss = {:.3f}, rwd_sum = {:.3f}, rwd_avg = {:.3f}".format(
                act_loss, crt_loss, rewards_sum, rewards_avg
            )
        )

    print("\n")


###


def main():
    env = gym.make(ENV_NAME)

    #

    # a2c_old = AdvantageActorCritic(ENV_NAME, N_MAX_EPISODE_STEPS)
    # run_agent(env, agent2)

    #

    actor_network = ActorNetwork(n_actions=env.action_space.n)
    critic_network = CriticNetwork()

    policy = NetworkPolicy(
        state_space=env.observation_space, action_space=env.action_space, network=actor_network
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
