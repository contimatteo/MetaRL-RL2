from typing import Any, Tuple

import time
import numpy as np
import gym

from loguru import logger

from memory import SequentialMemory
from __types import T_Action, T_State, T_Reward

###

N_EPISODES = 5
N_MAX_EPISODE_STEPS = 1000
N_EPISODE_STEP_SECONDS_DELAY = .3

env = gym.make("CartPole-v1")

###


class REINFORCE():

    def __init__(self) -> None:
        self.memory = SequentialMemory(N_MAX_EPISODE_STEPS)

    def act(self, state: T_State) -> T_Action:
        return env.action_space.sample()

    def remember(
        self, step: int, state: T_State, action: T_Action, reward: T_Reward, next_state: T_State
    ) -> None:
        self.memory.store(
            step, {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state
            }
        )


###


def main():
    agent = REINFORCE()

    next_state = None
    state, _ = env.reset(seed=42, return_info=True)
    env.render()

    for episode in range(N_EPISODES):
        logger.debug(f" > EPISODE = {episode}")

        agent.memory.reset()

        for step in range(N_MAX_EPISODE_STEPS):
            action = agent.act(state)

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
