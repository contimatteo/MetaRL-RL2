# import numpy as np

# import gym

# from gym import spaces
# from gym.utils import seeding

# ###

# class TwoArmedBanditsEnv(gym.Env):
#     """
#     The famous k-Armed Bandit Environment, implemented for the gym interface.
#     Initialization requires an array for the mean of each bandit,
#     as well as another array for the deviation from the mean for
#     each bandit. This arrays are then used to sample from the
#     distribution of a given bandit.
#     """
#     metadata = {'render.modes': ['human']}

#     def __init__(self, mean, stddev):
#         assert len(mean.shape) == 2
#         assert len(stddev.shape) == 2

#         super(TwoArmedBanditsEnv, self).__init__()
#         # Define action and observation space
#         self.num_bandits = mean.shape[1]
#         self.num_experiments = mean.shape[0]
#         self.action_space = spaces.Discrete(self.num_bandits)

#         # Theres one state only in the k-armed bandits problem
#         self.observation_space = spaces.Discrete(1)
#         self.mean = mean
#         self.stddev = stddev

#     def step(self, action):
#         # Sample from the specified bandit using it's reward distribution
#         # assert (action < self.num_bandits).all()
#         assert action < self.num_bandits

#         sampled_means = self.mean[np.arange(self.num_experiments), action]
#         sampled_stddevs = self.stddev[np.arange(self.num_experiments), action]

#         reward = np.random.normal(
#             loc=sampled_means, scale=sampled_stddevs, size=(self.num_experiments, )
#         )

#         # Return a constant state of 0. Our environment has no terminal state
#         observation, done, info = 0, False, dict()
#         return observation, reward, done, info

# ###

# ### Initialize the environment of our multi-armed bandit problem
# # num_experiments = 2
# # num_bandits = 8
# # means = np.random.normal(size=(num_experiments, num_bandits))
# # stdev = np.ones((num_experiments, num_bandits))
# # env = ArmedBanditsEnv(means, stdev)
