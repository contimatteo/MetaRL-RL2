from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.optimizers import adam_v2

from networks import ActorNetwork, CriticNetwork

from .base import BaseOldAgent

###


class AdvantageActorCritic(BaseOldAgent):
    """
    Advantage Actor-Critic (A2C)
    
    TODO:
     - {entropy_loss_coef} theory + application
     - {critic_loss_coef} theory + application
     - {_action_advantage_estimate} must be rewritten following "N-Step Advantage Estimate"
    """

    PY_NUMERIC_EPS = 1e-8

    def __init__(
        self,
        env_name: str,
        n_max_episode_steps: int,
        gamma: float = 0.99,
        action_advatange_n_steps: int = 1,
        standardize_action_advatange: bool = True,
        entropy_loss_coef: float = 1e-3,
        critic_loss_coef: float = 0.5,
        opt_gradient_clip_norm: float = 0.25,
        opt_actor_lr: float = 5e-4,
        opt_critic_lr: float = 5e-4,
    ) -> None:
        super(AdvantageActorCritic,
              self).__init__(env_name=env_name, n_max_episode_steps=n_max_episode_steps)

        self._gamma = gamma
        self._standardize_action_advatange = standardize_action_advatange
        self._entropy_loss_coef = entropy_loss_coef
        self._critic_loss_coef = critic_loss_coef
        self._opt_gradient_clip_norm = opt_gradient_clip_norm
        self._action_advatange_n_steps = action_advatange_n_steps

        self.actor_network = ActorNetwork(n_actions=self.env.action_space.n)
        self.critic_network = CriticNetwork()

        self.actor_network_optimizer = adam_v2.Adam(learning_rate=opt_actor_lr)
        self.critic_network_optimizer = adam_v2.Adam(learning_rate=opt_critic_lr)

    @property
    def name(self) -> str:
        return "A2C"

    #

    def _action_advantage_estimate(
        self, rewards: Any, state_values: Any, next_state_values: Any
    ) -> Any:
        """
        ### TD-Error (1-Step Advantage)
        `Aφ(s,a) = r(s,a,s′) + γVφ(s′) − Vφ(s)` \n
        if `s′` is terminal, then `Vφ(s′) ≐ 0`
        ### N-Step Advantage Estimate
        `Aφ(s,a) = ∑_{k=0..n−1} (γ^k * r_{t+k+1}) + (γ^n * Vφ(s_{t+n+1})) − Vφ(st)` \n
        """
        # advantages = tf.math.subtract(rewards, state_values)
        disc_next_state_values = tf.math.multiply(self._gamma, next_state_values)
        advantages = tf.math.subtract(rewards, state_values) + disc_next_state_values

        if self._standardize_action_advatange:
            advantages = (
                (advantages - tf.math.reduce_mean(advantages)) /
                (tf.math.reduce_std(advantages) + self.PY_NUMERIC_EPS)
            )

        return advantages

    def _critic_network_loss(self, advantages: Any, state_value: Any):
        # def _critic_network_loss(self, rewards: Any, state_value: Any):
        #     # return self._critic_loss_coef * mean_squared_error(advantage_estimates, state_value)
        #     return self._critic_loss_coef * mean_squared_error(rewards, state_value)
        return self._critic_loss_coef * mean_squared_error(advantages, state_value)

    def _actor_network_loss(self, actions_probs: Any, actions: Any, action_advantages: Any):
        policy_losses = []

        for step_actions_probs, action_taken, advantage_estimate in zip(
            actions_probs, actions, action_advantages.numpy()
        ):
            advantage = tf.constant(advantage_estimate)  ### exclude from gradient computation
            distribution = tfp.distributions.Categorical(
                probs=step_actions_probs + .000001, dtype=tf.float32
            )
            policy_loss = tf.math.multiply(distribution.log_prob(action_taken), advantage)
            policy_losses.append(policy_loss)

        # policy_loss = tf.reduce_mean(tf.stack(policy_losses))
        policy_loss = tf.reduce_sum(tf.stack(policy_losses))

        return -policy_loss

    #

    def _discounted_rewards(self, rewards: np.ndarray) -> np.ndarray:
        # discounted_rewards = []
        # reward_sum = 0

        # rewards = rewards.tolist()
        # rewards.reverse()

        # for r in rewards:
        #     reward_sum = r + self._gamma
        #     # reward_sum = r + self._gamma * reward_sum
        #     discounted_rewards.append(reward_sum)

        # discounted_rewards.reverse()

        # return np.array(discounted_rewards, dtype=np.float32)
        return rewards

    #

    def act(self, state: Any, random: bool = False) -> np.ndarray:
        ### DISCRETE (random)
        if random:
            return self.env.action_space.sample()

        ### DISCRETE (sample from probabilities distribution)
        probabilities = self.actor_network(np.array([state])).numpy()
        distribution = tfp.distributions.Categorical(
            probs=probabilities + .000001, dtype=tf.float32
        )
        return distribution.sample().numpy()

    def train(self, batch_size: int = None) -> None:
        steps_metrics = {"actor_nn_loss": [], "critic_nn_loss": [], "rewards": []}
        episode_metrics = {
            "steps": 0,
            "actor_nn_loss_avg": 0,
            "critic_nn_loss_avg": 0,
            "rewards_avg": 0,
            "rewards_sum": 0,
        }

        #

        episode = self.memory.all()

        _states = episode["states"]
        _next_states = episode["next_states"]
        _actions = episode["actions"]
        _rewards = episode["rewards"]
        _disc_rewards = self._discounted_rewards(_rewards)

        #

        actor_loss = []
        critic_loss = []

        _disc_rewards = tf.reshape(_disc_rewards, (len(_disc_rewards)))
        _disc_rewards = tf.cast(_disc_rewards, tf.float32)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            actions_probs = self.actor_network(_states, training=True)
            state_values = self.critic_network(_states, training=True)
            next_state_values = self.critic_network(_next_states, training=True)

            state_values = tf.reshape(state_values, (len(state_values)))
            next_state_values = tf.reshape(next_state_values, (len(next_state_values)))

            action_advantages = tf.stop_gradient(
                self._action_advantage_estimate(_disc_rewards, state_values, next_state_values)
            )

            actor_loss = self._actor_network_loss(actions_probs, _actions, action_advantages)
            critic_loss = self._critic_network_loss(action_advantages, state_values)

        #

        actor_grads = tape1.gradient(actor_loss, self.actor_network.trainable_variables)
        critic_grads = tape2.gradient(critic_loss, self.critic_network.trainable_variables)

        actor_grads, _ = tf.clip_by_global_norm(actor_grads, self._opt_gradient_clip_norm)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, self._opt_gradient_clip_norm)

        self.actor_network_optimizer.apply_gradients(
            zip(actor_grads, self.actor_network.trainable_variables)
        )
        self.critic_network_optimizer.apply_gradients(
            zip(critic_grads, self.critic_network.trainable_variables)
        )

        episode_metrics["steps"] = len(_states)
        steps_metrics["actor_nn_loss"] = actor_loss
        steps_metrics["critic_nn_loss"] = critic_loss
        steps_metrics["rewards"] = _rewards

        episode_metrics["actor_nn_loss_avg"] = np.mean(steps_metrics["actor_nn_loss"])
        episode_metrics["critic_nn_loss_avg"] = np.mean(steps_metrics["critic_nn_loss"])
        episode_metrics["actor_nn_loss_sum"] = np.sum(steps_metrics["actor_nn_loss"])
        episode_metrics["critic_nn_loss_sum"] = np.sum(steps_metrics["critic_nn_loss"])
        episode_metrics["rewards_avg"] = np.mean(steps_metrics["rewards"])
        episode_metrics["rewards_sum"] = np.sum(steps_metrics["rewards"])

        return episode_metrics
