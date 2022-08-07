import utils.env_setup

import gym

from loguru import logger
from progress.bar import Bar

from agents import ActorCritic

###

ENV_NAME = "MountainCar-v0"  # "CartPole-v1"
N_EPISODES = 10
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

    bar = Bar('Running Episodes ...', max=N_EPISODES)

    for episode in range(N_EPISODES):
        step = 0
        done = False

        state, _ = env.reset(seed=42, return_info=True)
        # env.render()

        agent.memory.reset()

        while not done:
            step += 1
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            # env.render()
            # logger.debug(f" > step = {step}, action = {action}, reward = {reward}, done = {done}")
            __sleep()

            agent.remember(step, state, action, reward, next_state, done)
            state = next_state

        episode_metrics, _ = agent.train()

        history.append(episode_metrics)
        bar.next()

    bar.finish()

    #

    print("\n\n\n")

    for episode_metrics in history:
        ep_actor_loss = episode_metrics["actor_network_loss"]
        ep_critic_loss = episode_metrics["critic_network_loss"]
        logger.debug(f" > actor_loss = {ep_actor_loss}, critic_loss = {ep_critic_loss}")
        # for step in range(episode_metrics["steps"]):
        #     st_actor_loss = episode_metrics["actor_network_loss"][step]
        #     st_critic_loss = episode_metrics["critic_network_loss"][step]
        #     logger.debug(f" > actor_loss = {st_actor_loss[0]}, critic_loss = {st_critic_loss[0]}")


###

if __name__ == "__main__":
    main()
