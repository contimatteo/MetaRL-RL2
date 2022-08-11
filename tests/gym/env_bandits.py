import gym
import numpy as np

from gym import spaces

###


class ArmedBanditsEnv(gym.Env):
    """
    The famous k-Armed Bandit Environment, implemented for the gym interface.
    Initialization requires an array for the mean of each bandit, 
    as well as another array for the deviation from the mean for 
    each bandit. This arrays are then used to sample from the 
    distribution of a given bandit.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, mean, stddev):
        assert len(mean.shape) == 2
        assert len(stddev.shape) == 2

        super(ArmedBanditsEnv, self).__init__()
        # Define action and observation space
        self.num_bandits = mean.shape[1]
        self.num_experiments = mean.shape[0]
        self.action_space = spaces.Discrete(self.num_bandits)

        # Theres one state only in the k-armed bandits problem
        self.observation_space = spaces.Discrete(1)
        self.mean = mean
        self.stddev = stddev

    def step(self, action):
        # Sample from the specified bandit using it's reward distribution
        assert action < self.num_bandits

        sampled_means = self.mean[np.arange(self.num_experiments), action]
        sampled_stddevs = self.stddev[np.arange(self.num_experiments), action]

        reward = np.random.normal(
            loc=sampled_means, scale=sampled_stddevs, size=(self.num_experiments, )
        )

        # Return a constant state of 0. Our environment has no terminal state
        observation, done, info = 0, False, dict()
        return observation, reward, done, info


###


def __simulation(num_experiments: int, num_bandits=int):
    means = np.random.normal(size=(num_experiments, num_bandits))
    stdev = np.ones((num_experiments, num_bandits))

    env = ArmedBanditsEnv(mean=means, stddev=stdev)
    env.reset()

    for _ in range(100):
        observation, reward, done, info = env.step(env.action_space.sample())


def test():
    __simulation(num_experiments=5, num_bandits=2)
    __simulation(num_experiments=6, num_bandits=3)
    __simulation(num_experiments=7, num_bandits=4)
    __simulation(num_experiments=8, num_bandits=5)
    __simulation(num_experiments=9, num_bandits=6)
    __simulation(num_experiments=10, num_bandits=7)
    __simulation(num_experiments=10, num_bandits=8)


###

if __name__ == "__main__":
    test()
