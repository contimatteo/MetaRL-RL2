import utils.env_setup

import gym

from loguru import logger

from agents import ActorCritic

###

ENV_NAME = "CartPole-v1"
N_EPISODES = 100
N_MAX_EPISODE_STEPS = 1000
N_EPISODE_STEP_SECONDS_DELAY = .3

env = gym.make(ENV_NAME)

###


def __env_reset():
    state, info = env.reset(seed=42, return_info=True)
    # env.render()
    return state, info


def __sleep():
    # time.sleep(N_EPISODE_STEP_SECONDS_DELAY)
    pass


###


def main():
    agent = ActorCritic(ENV_NAME, N_MAX_EPISODE_STEPS)

    history = []
    next_state = None
    state, _ = __env_reset()

    ### TRAIN

    for episode in range(N_EPISODES):
        logger.debug(f" > EPISODE = {episode}")

        agent.memory.reset()

        for step in range(N_MAX_EPISODE_STEPS):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            # env.render()
            # logger.debug(f" > step = {step}, action = {action}, reward = {reward}, done = {done}")
            __sleep()

            agent.remember(step, state, action, reward, next_state, done)
            state = next_state

            if done:
                episode_metrics = agent.train()
                history.append(episode_metrics)
                state, _ = __env_reset()
                break

    #

    print("\n\n\n")

    for episode_metrics in history:
        for step in range(episode_metrics["steps"]):
            actor_loss = episode_metrics["actor_network_loss"][step]
            critic_loss = episode_metrics["critic_network_loss"][step]
            logger.debug(f" > actor_loss = {actor_loss}, critic_loss = {critic_loss}")


###

if __name__ == "__main__":
    main()