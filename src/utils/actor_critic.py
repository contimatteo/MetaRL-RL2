from typing import Any, Union, Tuple

import numpy as np
import tensorflow as tf

###

T_ArrayOrTensor = Union[np.ndarray, tf.Tensor]

###

PY_NUMERIC_EPS = 1e-8

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
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        ### Generalized Advantage Estimate
        `δ = r(s,a,s′) + γVφ(s′) − Vφ(s)` \n
        `Aφ(s,a) = GAE(γ,λ)_{t} = ∑_l=0..∞} (γλ) δ_{t+l}` \n
        if `s′` is terminal, then `Vφ(s′) ≐ 0`
        """
        _rewards = rewards.numpy()
        not_dones = tf.math.subtract(1, dones)
        advantages = np.zeros_like(state_v)

        advantages[-1] = _rewards[-1] + (gamma * not_dones[-1] + next_state_v[-1]) - state_v[-1]

        for t in reversed(range(len(_rewards) - 1)):
            delta = _rewards[t] + (gamma * not_dones[t] * next_state_v[t]) - state_v[t]
            advantages[t] = delta + (gamma * gae_lambda * advantages[t + 1] * not_dones[t])

        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        return advantages, None

    @staticmethod
    def standardize(advantages: tf.Tensor) -> tf.Tensor:
        return (
            (advantages - tf.math.reduce_mean(advantages)) /
            (tf.math.reduce_std(advantages) + PY_NUMERIC_EPS)
        )
