import time
import gym

from loguru import logger

from agents import REINFORCE

###

N_EPISODES = 5
N_MAX_EPISODE_STEPS = 1000
N_EPISODE_STEP_SECONDS_DELAY = .3

env = gym.make("CartPole-v1")

###


def main():
    agent = REINFORCE(N_MAX_EPISODE_STEPS)

    next_state = None
    state, _ = env.reset(seed=42, return_info=True)
    env.render()

    for episode in range(N_EPISODES):
        logger.debug(f" > EPISODE = {episode}")

        agent.memory.reset()

        for step in range(N_MAX_EPISODE_STEPS):
            action = agent.act(env, state)

            next_state, reward, done, _ = env.step(action)
            env.render()

            logger.debug(f" > step = {step}, action = {action}, reward = {reward}, done = {done}")
            time.sleep(N_EPISODE_STEP_SECONDS_DELAY)

            agent.remember(step, state, action, reward, next_state)
            state = next_state

            if done:
                state, _ = env.reset(seed=42, return_info=True)
                env.render()
                break


###

if __name__ == "__main__":
    main()
