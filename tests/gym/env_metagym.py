import warnings

warnings.filterwarnings("ignore")

import gym
import numpy as np

import metagym.quadrotor
import metagym.quadrupedal
import metagym.navigator2d

###


def quadrupedal():
    env = gym.make('quadrupedal-v0', render=True, task="stairstair")
    env.reset()

    for i in range(100):
        action = env.action_space.sample()
        env.step(action)


def quadrotor():
    env = gym.make("quadrotor-v0", task="no_collision")
    env.reset()
    env.render()

    for i in range(100):
        action = env.action_space.sample()
        env.step(action)
        env.render()


def navigator2d():
    env = gym.make("navigator-wr-2D-v0", enable_render=True)

    env.set_task(env.sample_task())
    env.reset()

    for i in range(100):
        action = env.action_space.sample()
        env.step(action)


###

if __name__ == "__main__":
    quadrupedal()
    # quadrotor()
    # navigator2d()
