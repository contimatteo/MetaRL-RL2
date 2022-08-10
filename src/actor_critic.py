# pylint: disable=wrong-import-order, unused-import, consider-using-f-string
from typing import Union

import utils.env_setup

import gym
import numpy as np
import tensorflow as tf

from loguru import logger
from progress.bar import Bar

from agents import ActorCritic, AdvantageActorCritic
# from agents.a2xc import XAdvantageActorCritic

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


def __sleep():
    # time.sleep(N_EPISODE_STEP_SECONDS_DELAY)
    pass


###


def run_agent(agent: Union[ActorCritic, AdvantageActorCritic]):
    history = []
    step = None
    done = False
    next_state = None

    ### ENV

    env = gym.make(ENV_NAME)

    __seed(env)

    ### TRAIN

    print("\n")

    progbar = Bar('Running Episodes ...', max=N_EPISODES)

    for _ in range(N_EPISODES):
        step = 0
        done = False

        state, _ = env.reset(seed=42, return_info=True)
        ENV_RENDER and env.render()

        agent.memory.reset()

        while not done:
            step += 1
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            ENV_RENDER and env.render()
            # logger.debug(f" > step = {step}, action = {action}, reward = {reward}, done = {done}")
            __sleep()

            agent.remember(step, state, action, reward, next_state, done)
            state = next_state

        episode_metrics, _ = agent.train()

        history.append(episode_metrics)
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
    # agent1 = ActorCritic(ENV_NAME, N_MAX_EPISODE_STEPS)
    # run_agent(agent1)

    agent2 = AdvantageActorCritic(ENV_NAME, N_MAX_EPISODE_STEPS)
    run_agent(agent2)

    # agent3 = XAdvantageActorCritic(ENV_NAME, N_MAX_EPISODE_STEPS)
    # run_agent(agent3)


###

if __name__ == "__main__":
    main()
