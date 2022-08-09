import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.keras.optimizers import adam_v2

from models import ActorNetwork, CriticNetwork
from __types import T_Action, T_State, T_Reward, T_Rewards

from .base import Agent

###

AC_PARAMS_GAMMA = 0.99

###


class ActorCritic(Agent):

    def __init__(self, *args, **kwargs) -> None:
        super(ActorCritic, self).__init__(*args, **kwargs)

        self.gamma = AC_PARAMS_GAMMA

        self.actor_network = ActorNetwork(n_actions=self.env.action_space.n)
        self.critic_network = CriticNetwork()

        self.actor_network_optimizer = adam_v2.Adam(learning_rate=5e-5)
        self.critic_network_optimizer = adam_v2.Adam(learning_rate=5e-5)

    def configure(self, gamma: float = None):
        self.gamma = gamma if gamma is not None else self.gamma

    #

    def act(self, state: T_State, random: bool = False) -> T_Action:
        ### DISCRETE (random)
        if random:
            return self.env.action_space.sample()

        ### DISCRETE (sample from probabilities distribution)
        probabilities = self.actor_network(np.array([state])).numpy()
        distribution = tfp.distributions.Categorical(probs=probabilities, dtype=tf.float32)
        action_tensor = distribution.sample()
        return int(action_tensor.numpy()[0])

        ### CONTINOUOS (random)
        # probabilities = self.actor_network(np.array([state])).numpy()

    #

    def _action_advantage_estimate(
        self, reward: T_Reward, state: T_State, next_state: T_State, done: bool
    ) -> float:
        """
        TD-Advantage-Estimate (also called "TD-error" or "1-Step Advantage") \n
        `δ ← R + γv(S′, w) − v(S, w)` \n
        if S′ is terminal, then v(S′, w) ≐ 0
        """
        state_value = self.critic_network(state, training=True)
        next_state_value = self.critic_network(next_state, training=True) if not done else 0

        return reward + (self.gamma * next_state_value) - state_value

    def _actor_network_loss(self, actions_probs: np.ndarray, action: T_Action, td_error: float):
        """
        Negative of log probability of action taken multiplied
        by temporal difference used in q learning.
        """
        dist = tfp.distributions.Categorical(probs=actions_probs + .000001)
        return -1 * dist.log_prob(action) * td_error

    def _critic_network_loss(self, td_error: float):
        """
        2th-power operator is required for differentiability
        """
        return tf.math.pow(td_error, 2)

    def _discounted_rewards(self, rewards: T_Rewards) -> T_Rewards:
        return rewards

    #

    def train(self) -> None:
        """
        0. [Input] a differentiable policy parametrisation π(a|s, θ)
        1. [Input] a differentiable state-value function parametrisation v(s, w)
        2. [Parameters] step sizes αθ > 0 and αw > 0
        3. Initialise policy parameter θ ∈ ℝ^d′ and state-value weights w ∈ ℝ^d
        4. Loop forever (for each episode):
            1. Initialise S (ﬁrst state of episode)
            2. Loop while S is not terminal
                1. Select A using policy π
                2. Take action A, observe S′, R
                3. δ ← R + γv(S′, w) − v(S, w)
                4. w ← w + αw * δ * ∇v(S, w)
                5. θ ← θ + αθ * δ * ∇lnπ(A|S, θ)
                6. S ← S′
        """

        episode = self.memory.all()

        _states = episode["states"]
        _rewards = episode["rewards"]
        _actions = episode["actions"]
        _next_states = episode["next_states"]
        _done = episode["done"]

        steps_metrics = {"actor_nn_loss": [], "critic_nn_loss": [], "rewards": []}
        episode_metrics = {
            "steps": 0,
            "actor_nn_loss_avg": 0,
            "critic_nn_loss_avg": 0,
            "rewards_avg": 0,
            "rewards_sum": 0,
        }

        _discounted_rewards = self._discounted_rewards(_rewards)

        #

        for step in range(episode["steps"]):
            state = np.array([_states[step]])
            nxt_state = np.array([_next_states[step]])
            action, done = _actions[step], _done[step]
            reward, w_reward = _rewards[step], _discounted_rewards[step]

            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                actions_probs = self.actor_network(state, training=True)

                action_advantage = self._action_advantage_estimate(w_reward, state, nxt_state, done)
                actor_loss = self._actor_network_loss(actions_probs, action, action_advantage)
                critic_loss = self._critic_network_loss(action_advantage)

            actor_grads = tape1.gradient(actor_loss, self.actor_network.trainable_variables)
            critic_grads = tape2.gradient(critic_loss, self.critic_network.trainable_variables)

            self.actor_network_optimizer.apply_gradients(
                zip(actor_grads, self.actor_network.trainable_variables)
            )
            self.critic_network_optimizer.apply_gradients(
                zip(critic_grads, self.critic_network.trainable_variables)
            )

            episode_metrics["steps"] += 1
            steps_metrics["actor_nn_loss"].append(actor_loss)
            steps_metrics["critic_nn_loss"].append(critic_loss)
            steps_metrics["rewards"].append(reward)

        episode_metrics["actor_nn_loss_avg"] = np.mean(steps_metrics["actor_nn_loss"])
        episode_metrics["critic_nn_loss_avg"] = np.mean(steps_metrics["critic_nn_loss"])
        episode_metrics["rewards_avg"] = np.mean(steps_metrics["rewards"])
        episode_metrics["rewards_sum"] = np.sum(steps_metrics["rewards"])

        return episode_metrics, steps_metrics

    def test(self):
        pass
