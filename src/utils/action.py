import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions import Distribution

###


class ActionUtils():

    @staticmethod
    def coefficients_to_distribution(discrete: bool, coefficients: tf.Tensor) -> Distribution:

        if discrete:
            probabilities = coefficients.numpy()
            distribution = tfp.distributions.Categorical(logits=probabilities, dtype=tf.float32)
            return distribution
