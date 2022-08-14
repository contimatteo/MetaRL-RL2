from typing import Any, Union

import numpy as np
import tensorflow as tf

###

T_ArrayOrTensor = Union[np.ndarray, tf.Tensor]

###

PY_NUMERIC_EPS = 1e-8

# ###

# def TD1_error(
#     gamma, reward: float, state_value: float, next_state_value: float, done: int
# ) -> float:
#     """
#     ### TD-Error (1-Step Advantage)
#     `Aφ(s,a) = r(s,a,s′) + γVφ(s′) − Vφ(s)` \n
#     if `s′` is terminal, then `Vφ(s′) ≐ 0`
#     """
#     assert isinstance(gamma, float)
#     assert isinstance(reward, float)
#     assert isinstance(state_value, float)
#     assert isinstance(next_state_value, float)
#     assert isinstance(done, int) and (done == 0 or done == 1)
#     if done == 1:
#         next_state_value = 0
#     return reward + (gamma * next_state_value) - state_value

###

# def generalized_advantage_estimate(
#     gamma, gae_lambda, state_values, next_state_values, rewards, dones
# ):
#     assert isinstance(gamma, float)
#     assert isinstance(gae_lambda, float)
#     assert isinstance(state_values, tf.Tensor)
#     assert isinstance(next_state_values, tf.Tensor)
#     assert isinstance(rewards, np.ndarray)
#     assert isinstance(dones, np.ndarray)
#     assert state_values.shape == next_state_values.shape == rewards.shape == dones.shape
#     state_values = state_values.numpy()
#     next_state_values = next_state_values.numpy()
#     # rewards = np.reshape(rewards, -1)
#     # dones = 1 - np.reshape(dones, -1)
#     advantages = np.zeros_like(state_values)
#     assert state_values.shape == rewards.shape
#     ### advantage of the last timestep
#     advantages[-1] = TD1_error(
#         gamma, rewards[-1], state_values[-1], next_state_values[-1], dones[-1]
#     )
#     for t in reversed(range(len(rewards) - 1)):
#         delta = TD1_error(gamma, rewards[t], state_values[t], next_state_values[t], dones[t])
#         advantages[t] = delta + (gamma * gae_lambda * advantages[t + 1] * dones[t])
#     returns = tf.convert_to_tensor(advantages + state_values, dtype=tf.float32)
#     advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
#     ### TODO: try with and without the {tf.stop_gradient(...)} function because, if I'm not
#     ### wrong, it is useless, since we convert to `numpy` the {state_values} and
#     ### {next_state_values} input tensors.
#     ### INFO: stop gradient to avoid backpropagating through the advantage estimator
#     # return returns, advantages
#     # return tf.stop_gradient(returns), tf.stop_gradient(advantages)
#     return tf.stop_gradient(returns), tf.stop_gradient(advantages)

###

# def standardize_generalized_advantage_estimate(advantages):
#     return (
#         (advantages - tf.math.reduce_mean(advantages)) /
#         (tf.math.reduce_std(advantages) + PY_NUMERIC_EPS)
#     )

###


class AdvantageEstimateUtils():
    """
    ### A2C/A3C Advantage Estimate
    
    Advantage ActorCritic methods (A2C, A3C, GAE) approximate the advantage of an action: \n
    `Aφ(s,a)` is called the advantage estimate and should be equal to the real advantage in 
    expectation.
    """

    @staticmethod
    def MC(rewards: T_ArrayOrTensor, state_v: T_ArrayOrTensor) -> tf.Tensor:
        """
        ### MC Advantage Estimate
        `Aφ(s,a) = R(s,a) − Vφ(s)` \n
        the Q-value of the action being replaced by the actual return
        """

        # assert rewards.shape[0] == state_v.shape[0]

        return tf.math.subtract(rewards, state_v)

    @staticmethod
    def TD(
        gamma: float,
        rewards: T_ArrayOrTensor,
        state_v: T_ArrayOrTensor,
        next_state_v: T_ArrayOrTensor,
        dones: T_ArrayOrTensor,
    ) -> tf.Tensor:
        """
        ### TD Advantage Estimate or TD Error
        `Aφ(s,a) = r(s,a,s′) + γVφ(s′) − Vφ(s)` \n
        if `s′` is terminal, then `Vφ(s′) ≐ 0`
        """

        # assert isinstance(gamma, float)
        # assert rewards.shape[0] == state_v.shape[0] == dones.shape[0]
        # assert state_v.shape == next_state_v.shape

        not_dones = tf.math.subtract(1, dones)
        _next_state_v = tf.math.multiply(not_dones, next_state_v)  ### Vφ(s′) ≐ 0 if s' is terminal

        _expr1 = tf.math.subtract(rewards, state_v)  ### r(s,a,s′) − Vφ(s)
        _expr2 = tf.math.multiply(gamma, _next_state_v)  ### γVφ(s′)
        td = _expr1 + _expr2

        return tf.cast(td, dtype=tf.float32)

    @staticmethod
    def NStep() -> tf.Tensor:
        """
        ### N-Step Advantage Estimate
        `Aφ(s,a) = ∑_{k=0..n−1} (γ^k * r_{t+k+1}) + (γ^n * Vφ(s_{t+n+1})) − Vφ(st)` \n
        if `s′` is terminal, then `Vφ(s_{t+n+1} ≐ 0`
        """

        raise NotImplementedError

    @staticmethod
    def GAE(
        gamma: float,
        gae_lambda: float,
        rewards: T_ArrayOrTensor,
        state_v: T_ArrayOrTensor,
        next_state_v: T_ArrayOrTensor,
        dones: T_ArrayOrTensor,
    ) -> tf.Tensor:
        """
        ### Generalized Advantage Estimate
        `δ = r(s,a,s′) + γVφ(s′) − Vφ(s)` \n
        `Aφ(s,a) = GAE(γ,λ)_{t} = ∑_l=0..∞} (γλ) δ_{t+l}` \n
        if `s′` is terminal, then `Vφ(s′) ≐ 0`
        """

        # assert isinstance(gamma, float)
        # assert isinstance(gae_lambda, float)
        # assert rewards.shape[0] == state_v.shape[0] == dones.shape[0]
        # assert state_v.shape == next_state_v.shape

        _rewards = rewards.numpy()
        not_dones = tf.math.subtract(1, dones)
        advantages = np.zeros_like(state_v)

        advantages[-1] = _rewards[-1] + (gamma * not_dones[-1] + next_state_v[-1]) - state_v[-1]

        for t in reversed(range(len(_rewards) - 1)):
            delta = _rewards[t] + (gamma * not_dones[t] * next_state_v[t]) - state_v[t]
            advantages[t] = delta + (gamma * gae_lambda * advantages[t + 1] * not_dones[t])

        return tf.convert_to_tensor(advantages, dtype=tf.float32)

    @staticmethod
    def standardize(advantages: tf.Tensor) -> tf.Tensor:
        return (
            (advantages - tf.math.reduce_mean(advantages)) /
            (tf.math.reduce_std(advantages) + PY_NUMERIC_EPS)
        )
