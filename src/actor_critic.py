# pylint: disable=wrong-import-order, unused-import, consider-using-f-string
import utils.env_setup

import gym

from loguru import logger
from progress.bar import Bar

from agents import ActorCritic

###

ENV_RENDER = False
ENV_NAME = "LunarLander-v2"  # CartPole-v1 | MountainCar-v0

N_EPISODES = 1000
N_MAX_EPISODE_STEPS = 1000
N_EPISODE_STEP_SECONDS_DELAY = .3

###


def __sleep():
    # time.sleep(N_EPISODE_STEP_SECONDS_DELAY)
    pass


###


def main():
    agent = ActorCritic(ENV_NAME, N_MAX_EPISODE_STEPS)

    history = []
    step = None
    done = False
    next_state = None

    ### ENV

    env = gym.make(ENV_NAME)

    # print("\n\n")
    # logger.debug(f" > env = {ENV_NAME}")
    # logger.debug(f" > action_space = {env.action_space}")
    # logger.debug(f" > observation_space = {env.observation_space}")
    # print("\n\n")

    ### TRAIN

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

    print("\n\n\n")

    for episode_metrics in history:
        act_loss = round(episode_metrics["actor_nn_loss_avg"], 4)
        crt_loss = round(episode_metrics["critic_nn_loss_avg"], 4)
        rewards_sum = episode_metrics["rewards_sum"]
        rewards_avg = episode_metrics["rewards_avg"]
        logger.debug(
            "a_loss = {}, c_loss = {}, rewards_sum = {}, rewards_avg = {}".format(
                act_loss, crt_loss, rewards_sum, rewards_avg
            )
        )

    print("\n\n\n")

    #

    env.close()


###

if __name__ == "__main__":
    main()
