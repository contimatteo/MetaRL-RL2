from typing import Any

import numpy as np
import tensorflow as tf

from policies import Policy
from networks import ActorNetwork, CriticNetwork

from .a2c import A2C

###


class A3C(A2C):
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
        standardize_advantage_estimate: bool = True,
        critic_loss_coef: float = 0.5,
        opt_gradient_clip_norm: float = 0.25,
        opt_actor_lr: float = 1e-4,
        opt_critic_lr: float = 5e-4,
        entropy_loss_coef: float = 1e-3,
        gae_lambda: float = 0.9,
    ) -> None:
        super(A3C, self).__init__(
            n_max_episode_steps=n_max_episode_steps,
            policy=policy,
            actor_network=actor_network,
            critic_network=critic_network,
            gamma=gamma,
            standardize_advantage_estimate=standardize_advantage_estimate,
            critic_loss_coef=critic_loss_coef,
            opt_gradient_clip_norm=opt_gradient_clip_norm,
            opt_actor_lr=opt_actor_lr,
            opt_critic_lr=opt_critic_lr,
        )

        self._gae_lambda = gae_lambda
        self._entropy_loss_coef = entropy_loss_coef

    #

    @property
    def name(self) -> str:
        return "A3C"

    #

    def _advantage_estimates(self, rewards: Any, state_v: Any, next_state_v: Any, done: Any) -> Any:
        """
        Generalized Advantage Estimation
        """
        gamma = self._gamma
        gae_lambda = self._gae_lambda

        rewards = rewards.numpy()
        not_dones = 1 - np.reshape(done, -1)
        advantages = np.zeros_like(state_v)

        advantages[-1] = rewards[-1] + (gamma * not_dones[-1] + next_state_v[-1]) - state_v[-1]

        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + (gamma * not_dones[t] * next_state_v[t]) - state_v[t]
            advantages[t] = delta + (gamma * gae_lambda * advantages[t + 1] * not_dones[t])

        # returns = tf.convert_to_tensor(advantages + state_v, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        ### normalization taken care by the underlying algo
        ### stop gradient to avoid backpropagating through the advantage estimator
        # return tf.stop_gradient(returns), tf.stop_gradient(advantages)

        return self._standardize_advantage_estimates(advantages)
