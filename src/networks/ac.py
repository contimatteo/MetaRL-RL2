from typing import Tuple

import gym

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input

from networks.layers import A_HeadLayer, C_HeadLayer
from networks.layers import AC_EncoderLayer, AC_BackboneLayer

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
