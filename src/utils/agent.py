# import numpy as np
# import tensorflow as tf

# ###

# PY_NUMERIC_EPS = 1e-8

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

# def standardize_generalized_advantage_estimate(advantages):
#     return (
#         (advantages - tf.math.reduce_mean(advantages)) /
#         (tf.math.reduce_std(advantages) + PY_NUMERIC_EPS)
#     )
