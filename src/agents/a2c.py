from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.keras.losses import mean_squared_error, MeanSquaredError
from tensorflow.python.keras.optimizers import adam_v2

from models import ActorNetwork, CriticNetwork
from __types import T_Action, T_Actions, T_State, T_Reward, T_Rewards

from .base import Agent

###


class AdvantageActorCritic(Agent):
    """
    Advantage Actor-Critic (A2C)
    
    TODO:
     - {actor_entropy_coef} theory + application
     - {critic_loss_coef} theory + application
     - {_action_advantage_estimate} must be rewritten following "N-Step Advantage Estimate"
    """

    def __init__(
        self,
        env_name: str,
        n_max_episode_steps: int,
        gamma: float = 0.99,
        action_advatange_n_steps: int = 1,
        actor_entropy_coef: float = 1e-3,
        opt_gradient_clip_norm: float = 0.25,
        opt_actor_lr: float = 1e-3,
        opt_critic_lr: float = 1e-3,
    ) -> None:
        super(AdvantageActorCritic,
              self).__init__(env_name=env_name, n_max_episode_steps=n_max_episode_steps)

        self._gamma = gamma
        self._actor_entropy_coef = actor_entropy_coef
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

    def _action_advantage_estimate(self, discounted_rewards: T_Reward, state_values: Any) -> float:
        """
        ### N-Step Advantage Estimate
        `Aφ(s,a) = ∑_{k=0..n−1} (γ^k * r_{t+k+1}) + (γ^n * Vφ(s_{t+n+1})) − Vφ(st)` \n
        """
        return tf.math.subtract(discounted_rewards, state_values)

    def _critic_network_loss(self, discounted_rewards: T_Rewards, state_value: Any):
        critic_loss_coef = 0.5
        return critic_loss_coef * mean_squared_error(discounted_rewards, state_value)

    def _actor_network_loss(self, actions_probs: Any, actions: Any, action_advantages: Any):
        policy_losses = []
        entropy_losses = []

        for step_actions_probs, action_taken, advantage_estimate in zip(
            actions_probs, actions, action_advantages.numpy()
        ):
            advantage = tf.constant(advantage_estimate)  ### exclude from gradient computation
            #
            distribution = tfp.distributions.Categorical(
                probs=step_actions_probs + .000001, dtype=tf.float32
            )
            action_taken_prob = distribution.prob(action_taken)
            action_taken_log_prob = distribution.log_prob(action_taken)
            #
            _policy_loss = tf.math.multiply(action_taken_log_prob, advantage)
            _entropy_loss = tf.math.negative(
                tf.math.multiply(action_taken_prob, action_taken_log_prob)
            )
            #
            policy_losses.append(_policy_loss)
            entropy_losses.append(_entropy_loss)

        policy_loss = tf.reduce_mean(tf.stack(policy_losses))
        entropy_loss = self._actor_entropy_coef * tf.reduce_mean(tf.stack(entropy_losses))

        return -policy_loss - entropy_loss

    #

    def _discounted_rewards(self, rewards: np.ndarray) -> np.ndarray:
        discounted_rewards = []
        reward_sum = 0

        rewards = rewards.tolist()
        rewards.reverse()

        for r in rewards:
            reward_sum = r + self._gamma * reward_sum
            discounted_rewards.append(reward_sum)

        discounted_rewards.reverse()

        return np.array(discounted_rewards, dtype=np.float32)

    #

    def act(self, state: T_State, random: bool = False) -> T_Action:
        ### DISCRETE (random)
        if random:
            return self.env.action_space.sample()

        ### DISCRETE (sample from probabilities distribution)
        probabilities = self.actor_network(np.array([state])).numpy()
        distribution = tfp.distributions.Categorical(
            probs=probabilities + .000001, dtype=tf.float32
        )
        action_tensor = distribution.sample()
        return int(action_tensor.numpy()[0])

    def train(self) -> None:
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
        _actions = episode["actions"]
        _rewards = episode["rewards"]
        _discounted_rewards = self._discounted_rewards(_rewards)

        #

        actor_loss = []
        critic_loss = []

        _discounted_rewards = tf.reshape(_discounted_rewards, (len(_discounted_rewards)))

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            actions_probs = self.actor_network(_states, training=True)
            state_values = self.critic_network(_states, training=True)

            state_values = tf.reshape(state_values, (len(state_values)))

            action_advantages = self._action_advantage_estimate(_discounted_rewards, state_values)

            actor_loss = self._actor_network_loss(actions_probs, _actions, action_advantages)
            critic_loss = self._critic_network_loss(_discounted_rewards, state_values)

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
        episode_metrics["rewards_avg"] = np.mean(steps_metrics["rewards"])
        episode_metrics["rewards_sum"] = np.sum(steps_metrics["rewards"])

        return episode_metrics, steps_metrics
