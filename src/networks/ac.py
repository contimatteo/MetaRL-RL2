from typing import Tuple

import gym

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input

from networks.layers import A_HeadLayer, C_HeadLayer
from networks.layers import AC_EncoderLayer, AC_BackboneLayer

###


def ActorCriticNetworks(
    obs_space: gym.Space,
    action_space: gym.Space,
    shared_backbone: bool,
) -> Tuple[Model, Model]:
    discrete = isinstance(action_space, gym.spaces.discrete.Discrete)

    input_shape = obs_space.shape if len(obs_space.shape) > 0 else (1, )
    n_actions = action_space.n if discrete else action_space.shape[0]

    ### input
    l_input = Input(shape=input_shape)
    ### encoder
    l_encoder = AC_EncoderLayer()
    ### backbone
    if shared_backbone:
        l_backbone_shared = AC_BackboneLayer()
    else:
        l_backbone_a = AC_BackboneLayer()
        l_backbone_c = AC_BackboneLayer()
    ### head
    l_actor_head = A_HeadLayer(n_actions, discrete=discrete)
    l_critic_head = C_HeadLayer()

    #

    ### encoder
    out_encoder = l_encoder(l_input)

    ### backbone
    if shared_backbone:
        out_backbone = l_backbone_shared(out_encoder)
        out_backbone_a = out_backbone
        out_backbone_c = out_backbone
    else:
        out_backbone_a = l_backbone_a(out_encoder)
        out_backbone_c = l_backbone_c(out_encoder)

    ### heads
    out_actor = l_actor_head(out_backbone_a)
    out_critic = l_critic_head(out_backbone_c)

    #

    Actor = Model(inputs=l_input, outputs=out_actor)
    Critic = Model(inputs=l_input, outputs=out_critic)

    return Actor, Critic
