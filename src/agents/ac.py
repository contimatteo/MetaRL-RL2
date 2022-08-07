from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import gradient_descent_v2

from __types import T_Action, T_State, T_Reward

from .baseline import Agent

###

AC_PARAMS_GAMMA = 0.99

###


class CriticNetwork(Model):

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

        self.l1 = Dense(512, activation='relu')
        self.l2 = Dense(512, activation='relu')
        self.out = Dense(1, activation='linear')

    def call(self, input_data: Any, training=None, mask=None):
        x = self.l1(input_data)
        x = self.l2(x)
        x = self.out(x)
        return x

    def loss_eval(self, td_error) -> Any:
        ### operator is required for differentiability
        return td_error**2


class ActorNetwork(Model):

    def __init__(self, n_actions: int, *args: Any, **kwargs: Any):
        super().__init__()

        assert isinstance(n_actions, int)

        self.l1 = Dense(512, activation='relu')
        self.l2 = Dense(512, activation='relu')
        self.out = Dense(n_actions, activation='softmax')  # discrete
        # self.out = Dense(2, activation='linear')  # continuous

    def call(self, input_data: Any, training=None, mask=None):
        x = self.l1(input_data)
        x = self.l2(x)
        x = self.out(x)
        return x

    def loss_eval(self, td_error, action, actions_probs_pred):
        """
        Negative of log probability of action taken multiplied by temporal difference used in q learning.
        """
        dist = tfp.distributions.Categorical(probs=actions_probs_pred, dtype=tf.float32)
        return -1 * dist.log_prob(action) * td_error


###


class ActorCritic(Agent):

    def __init__(self, *args, **kwargs) -> None:
        # super(Agent, self).__init__(*args, **kwargs)
        super(ActorCritic, self).__init__(*args, **kwargs)

        self.gamma = AC_PARAMS_GAMMA

        self.actor_network = ActorNetwork(n_actions=self.env.action_space.n)
        self.critic_network = CriticNetwork()

        self.actor_network_optimizer = gradient_descent_v2.SGD(learning_rate=5e-4)
        self.critic_network_optimizer = gradient_descent_v2.SGD(learning_rate=5e-4)

    def configure(self, gamma=AC_PARAMS_GAMMA):
        self.gamma = gamma

    #

    def act(self, state: T_State, random: bool = False) -> T_Action:
        ### TODO: handle continuous action space
        ### ...

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

    def __compute_TD_error(
        self, reward: float, state_value: float, next_state_value: float, done: bool
    ) -> float:
        """
         δ ← R + γv(S′, w) − v(S, w)
         (if S′ is terminal, then v(S′, w) ≐ 0)
        """
        ### TODO: print values to check the formula
        if done:
            next_state_value = 0
        return reward + (self.gamma * next_state_value) - state_value

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

        episode_metrics = {"steps": 0, "actor_network_loss": 0, "critic_network_loss": 0}
        steps_metrics = {"actor_network_loss": [], "critic_network_loss": []}

        for step in range(episode["steps"]):
            state = np.array([_states[step]])
            next_state = np.array([_next_states[step]])
            action, reward, done = _actions[step], _rewards[step], _done[step]

            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                actions_probs_pred = self.actor_network(state, training=True)
                state_value_pred = self.critic_network(state, training=True)
                next_state_value_pred = self.critic_network(next_state, training=True)

                td_error = self.__compute_TD_error(
                    reward, state_value_pred, next_state_value_pred, done
                )

                actor_loss = self.actor_network.loss_eval(td_error, action, actions_probs_pred)
                critic_loss = self.critic_network.loss_eval(td_error)

                episode_metrics["steps"] += 1
                steps_metrics["actor_network_loss"].append(actor_loss)
                steps_metrics["critic_network_loss"].append(critic_loss)

            actor_network_gradients = tape1.gradient(
                actor_loss, self.actor_network.trainable_variables
            )
            critic_network_gradients = tape2.gradient(
                critic_loss, self.critic_network.trainable_variables
            )

            self.actor_network_optimizer.apply_gradients(
                zip(actor_network_gradients, self.actor_network.trainable_variables)
            )
            self.critic_network_optimizer.apply_gradients(
                zip(critic_network_gradients, self.critic_network.trainable_variables)
            )

        episode_metrics["actor_network_loss"] = np.mean(steps_metrics["actor_network_loss"])
        episode_metrics["critic_network_loss"] = np.mean(steps_metrics["critic_network_loss"])

        return episode_metrics, steps_metrics

    def test(self):
        pass
