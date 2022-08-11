from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.optimizers import adam_v2

from policies import Policy
from networks import ActorNetwork, CriticNetwork

from .agent import Agent

###

PY_NUMERIC_EPS = 1e-8

###


class A2C(Agent):
    """
    Advantage Actor-Critic (A2C)
    TODO:
     - {entropy_loss_coef} theory + application
     - {critic_loss_coef} theory + application
     - {_action_advantage_estimate} must be rewritten following "N-Step Advantage Estimate"
    """

    def __init__(
        self,
        n_max_episode_steps: int,
        policy: Policy,
        actor_network: ActorNetwork,
        critic_network: CriticNetwork,
        gamma: float = 0.99,
        standardize_action_advatange: bool = True,
        entropy_loss_coef: float = 1e-3,
        critic_loss_coef: float = 0.5,
        opt_gradient_clip_norm: float = 0.25,
        opt_actor_lr: float = 5e-4,
        opt_critic_lr: float = 5e-4,
    ) -> None:
        super(A2C, self).__init__(n_max_episode_steps=n_max_episode_steps, policy=policy)

        self._gamma = gamma
        self._critic_loss_coef = critic_loss_coef
        self._entropy_loss_coef = entropy_loss_coef
        self._opt_gradient_clip_norm = opt_gradient_clip_norm
        self._standardize_action_advatange = standardize_action_advatange

        self.actor_network = actor_network  # ActorNetwork(n_actions=self.env.action_space.n)
        self.critic_network = critic_network  # CriticNetwork()

        self.actor_network_optimizer = adam_v2.Adam(learning_rate=opt_actor_lr)
        self.critic_network_optimizer = adam_v2.Adam(learning_rate=opt_critic_lr)

    #

    @property
    def name(self) -> str:
        return "A2C"

    #

    def act(self, state: np.ndarray) -> int:
        assert isinstance(state, np.ndarray)

        ### flatten input observation
        state = state.flatten()  ###
        ### reshape in order to match network `batch` dimension
        state = np.expand_dims(state, axis=0)  ### (x,) -> (1, x)

        action_foreach_state = self.policy.act(state)

        ### convert `numpy.int` type to native python type
        action = action_foreach_state[0].item()

        return action

    def train(self, batch_size: int) -> Any:
        ep_metrics = []

        #

        ep_data = self.memory.to_tf_dataset()

        for ep_data_batch in ep_data.batch(batch_size):
            _states = ep_data_batch["states"]
            _rewards = ep_data_batch["rewards"]
            _actions = ep_data_batch["actions"]
            _next_states = ep_data_batch["next_states"]
            _done = ep_data_batch["done"]

            assert _states.shape[0] == _rewards.shape[0] == _actions.shape[0]
            assert _states.shape[0] == _next_states.shape[0] == _done.shape[0]

            print("")
            print("_states =", _states.shape)
            print("")

        #

        return ep_metrics
