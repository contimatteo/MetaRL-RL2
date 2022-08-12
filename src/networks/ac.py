from typing import Any, Tuple, Callable

import gym
import tensorflow as tf

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Layer, Dense, Flatten, Input

###


def AC_BackboneLayer() -> Callable[[tf.Tensor], Layer]:

    def backbone(x: tf.Tensor) -> Layer:
        x = Dense(512, activation="relu", kernel_initializer='he_uniform')(x)
        x = Dense(512, activation="relu", kernel_initializer='he_uniform')(x)
        return x

    return backbone


def AC_EncoderLayer() -> Callable[[tf.Tensor], Layer]:

    def encoder(x: tf.Tensor) -> Layer:
        x = Flatten()(x)
        return x

    return encoder


def A_HeadLayer(n_actions: int, discrete: bool) -> Callable[[tf.Tensor], Layer]:
    activation = "softmax" if discrete else "linear"

    def head(x: tf.Tensor) -> Layer:
        x = Dense(n_actions, activation=activation, kernel_initializer='he_uniform')(x)
        return x

    return head


def C_HeadLayer() -> Callable[[tf.Tensor], Layer]:

    def head(x: tf.Tensor) -> Layer:
        x = Dense(1, activation='linear')(x)
        return x

    return head


###


def ActorCriticNetworks(obs_space: gym.Space, action_space: gym.Space) -> Tuple[Model, Model]:
    ### TODO: support also 'continuous' action space
    assert isinstance(action_space, gym.spaces.discrete.Discrete)
    discrete = True

    input_shape = obs_space.shape

    ### input
    l_input = Input(shape=input_shape)
    ### encoder
    l_encoder = AC_EncoderLayer()
    ### backbone
    l_shared_backbone = AC_BackboneLayer()
    ### head
    l_actor_head = A_HeadLayer(action_space.n, discrete=discrete)
    l_critic_head = C_HeadLayer()

    #

    out_encoder = l_encoder(l_input)
    out_backbone = l_shared_backbone(out_encoder)
    out_actor = l_actor_head(out_backbone)
    out_critic = l_critic_head(out_backbone)

    #

    Actor = Model(inputs=l_input, outputs=out_actor)
    Critic = Model(inputs=l_input, outputs=out_critic)

    return Actor, Critic


###


class CriticNetwork(Model):

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

        self.flat = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(512, activation='relu')
        self.out = Dense(1, activation='linear')

    def call(self, input_data: Any, training=None, mask=None):
        x = self.flat(input_data)
        x = self.d1(x)
        x = self.d2(x)
        x = self.out(x)
        return x


class ActorNetwork(Model):

    def __init__(self, n_actions: int, *args: Any, **kwargs: Any):
        super().__init__()

        assert isinstance(n_actions, int)

        self.flat = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(512, activation='relu')
        self.out = Dense(n_actions, activation='softmax')  # discrete
        # self.out = Dense(n_actions, activation='linear')  # continuous

    def call(self, input_data: Any, training=None, mask=None):
        x = self.flat(input_data)
        x = self.d1(x)
        x = self.d2(x)
        x = self.out(x)
        return x