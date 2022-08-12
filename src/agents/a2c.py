from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.keras import Model
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.optimizers import adam_v2

from policies import Policy

from .agent import Agent

###


class A2C(Agent):

    PY_NUMERIC_EPS = 1e-8

    def __init__(
        self,
        n_max_episode_steps: int,
        policy: Policy,
        actor_network: Model,
        critic_network: Model,
        gamma: float = 0.99,
        standardize_advantage_estimate: bool = True,
        critic_loss_coef: float = 0.5,
        opt_gradient_clip_norm: float = 0.25,
        opt_actor_lr: float = 1e-4,
        opt_critic_lr: float = 5e-4,
    ) -> None:
        super(A2C, self).__init__(n_max_episode_steps=n_max_episode_steps, policy=policy)

        self._gamma = gamma
        self._critic_loss_coef = critic_loss_coef
        self._opt_gradient_clip_norm = opt_gradient_clip_norm
        self._standardize_advantage_estimate = standardize_advantage_estimate

        self.actor_network = actor_network  # ActorNetwork(n_actions=self.env.action_space.n)
        self.critic_network = critic_network  # CriticNetwork()

        self.actor_network_optimizer = adam_v2.Adam(learning_rate=opt_actor_lr)
        self.critic_network_optimizer = adam_v2.Adam(learning_rate=opt_critic_lr)

    #

    @property
    def name(self) -> str:
        return "A2C"

    def _discount_rewards(self, rewards: np.ndarray) -> np.ndarray:
        # discounted_rewards, reward_sum = [], 0
        # rewards = rewards.tolist()
        # rewards.reverse()
        # for r in rewards:
        #     reward_sum = r + self._gamma * reward_sum
        #     discounted_rewards.append(reward_sum)
        # discounted_rewards.reverse()
        # rewards = np.array(discounted_rewards, dtype=np.float32)
        return rewards

    #

    ### TODO: must be rewritten following "N-Step Advantage Estimate"
    def _advantage_estimates(self, rewards: Any, state_v: Any, next_state_v: Any, done: Any) -> Any:
        """
        ### TD-Error (1-Step Advantage)
        `Aφ(s,a) = r(s,a,s′) + γVφ(s′) − Vφ(s)` \n
        if `s′` is terminal, then `Vφ(s′) ≐ 0`
        ### N-Step Advantage Estimate
        `Aφ(s,a) = ∑_{k=0..n−1} (γ^k * r_{t+k+1}) + (γ^n * Vφ(s_{t+n+1})) − Vφ(st)` \n
        """
        _advantages_expr1 = tf.math.subtract(rewards, state_v)  ### r(s,a,s′) − Vφ(s)
        _advantages_expr2 = tf.math.multiply(self._gamma, next_state_v)  ### γVφ(s′)
        advantages = _advantages_expr1 + _advantages_expr2

        return self._standardize_advantage_estimates(advantages)

    def _standardize_advantage_estimates(self, advantages: Any) -> Any:
        if not self._standardize_advantage_estimate:
            return advantages

        return (
            (advantages - tf.math.reduce_mean(advantages)) /
            (tf.math.reduce_std(advantages) + self.PY_NUMERIC_EPS)
        )

    def _critic_network_loss(self, advantages: Any, state_value: Any):
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

    def act(self, state: np.ndarray) -> np.ndarray:
        return self.policy.act(state)

    def train(self, batch_size: int) -> Any:
        steps_metrics = {
            "actor_nn_loss": [],
            "critic_nn_loss": [],
            "rewards": [],
        }
        episode_metrics = {
            "steps": 0,
            "actor_nn_loss_avg": 0,
            "critic_nn_loss_avg": 0,
            "rewards_avg": 0,
            "rewards_sum": 0,
        }

        #

        ep_data = self.memory.to_tf_dataset()

        ### TODO: with or without shuffling?
        # ep_data_shuffled = ep_data.shuffle(ep_data.cardinality())
        # for ep_data_batch in ep_data_shuffled.batch(batch_size):
        for ep_data_batch in ep_data.batch(batch_size):
            _states = ep_data_batch["states"]
            _rewards = ep_data_batch["rewards"]
            _actions = ep_data_batch["actions"]
            _next_states = ep_data_batch["next_states"]
            _done = ep_data_batch["done"]

            _disc_rewards = self._discount_rewards(_rewards)
            _disc_rewards = tf.cast(_disc_rewards, tf.float32)

            assert _states.shape[0] == _disc_rewards.shape[0] == _actions.shape[0]
            assert _states.shape[0] == _next_states.shape[0] == _done.shape[0]

            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                actions_probs = self.actor_network(_states, training=True)
                state_values = self.critic_network(_states, training=True)
                next_state_values = self.critic_network(_next_states, training=True)

                state_values = tf.reshape(state_values, (len(state_values)))
                next_state_values = tf.reshape(next_state_values, (len(next_state_values)))

                action_advantages = tf.stop_gradient(
                    self._advantage_estimates(
                        _disc_rewards, state_values, next_state_values, _done
                    )
                )

                actor_loss = self._actor_network_loss(actions_probs, _actions, action_advantages)
                critic_loss = self._critic_network_loss(action_advantages, state_values)

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

            steps_metrics["actor_nn_loss"].append(actor_loss)
            steps_metrics["critic_nn_loss"].append(critic_loss)
            steps_metrics["rewards"] += _rewards.numpy().tolist()

        episode_metrics["steps"] = ep_data.cardinality().numpy()
        episode_metrics["actor_nn_loss_avg"] = np.mean(steps_metrics["actor_nn_loss"])
        episode_metrics["critic_nn_loss_avg"] = np.mean(steps_metrics["critic_nn_loss"])
        episode_metrics["actor_nn_loss_sum"] = np.sum(steps_metrics["actor_nn_loss"])
        episode_metrics["critic_nn_loss_sum"] = np.sum(steps_metrics["critic_nn_loss"])
        episode_metrics["rewards_avg"] = np.mean(steps_metrics["rewards"])
        episode_metrics["rewards_sum"] = np.sum(steps_metrics["rewards"])

        return episode_metrics
