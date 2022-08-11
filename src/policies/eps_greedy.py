from typing import Any

import random

from .policy import Policy
from .random import RandomPolicy

###

EPS_VAL = 0.1
EPS_MIN_VAL = 0.01
EPS_DECAY_VAL = 0.99

###


class EpsilonGreedyPolicy(Policy):

    def __init__(
        self, policy: Policy, epsilon=EPS_VAL, eps_decay=EPS_DECAY_VAL, eps_min_value=EPS_MIN_VAL
    ):
        super().__init__(policy.state_space, policy.action_space)

        self.epsilon = epsilon
        self.epsilon_decay = eps_decay
        self.epsilon_min = eps_min_value

        self.policy = policy
        self.policy_random = RandomPolicy(policy.state_space, policy.action_space)

    def exec_epsilon_decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    #

    def _act(self, obs: Any, **kwargs):
        eps_rand = random.random()  ### [0, 1]

        if eps_rand <= self.epsilon:
            return self.policy_random.act(obs, **kwargs)
        else:
            return self.policy.act(obs, **kwargs)
