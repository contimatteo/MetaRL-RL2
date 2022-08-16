from typing import Tuple, Optional

import gym

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Concatenate

from networks.layers import A_HeadLayer, C_HeadLayer
from networks.layers import AC_EncoderLayer
from networks.layers import AC_BackboneLayer
from networks.layers import AC_MetaMemoryLayer

###


def MetaActorCriticNetworks(
    obs_space: gym.Space,
    action_space: gym.Space,
    shared_backbone: bool,
) -> Tuple[Model, Model]:
    discrete = isinstance(action_space, gym.spaces.discrete.Discrete)

    obs_shape = obs_space.shape if len(obs_space.shape) > 0 else (1, )
    n_actions = action_space.n if discrete else action_space.shape[0]

    ### input
    input_obs = Input(shape=obs_shape, name="Input_Observations")
    input_prev_action = Input(shape=(n_actions, ), name="Input_PreviousAction")
    input_prev_reward = Input(shape=(1, ), name="Input_PreviousReward")
    ### encoder
    l_encoder = AC_EncoderLayer()
    ### meta-memory
    l_memory = AC_MetaMemoryLayer(name="MetaMemory")
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
    out_encoder = l_encoder(input_obs)

    ### memory
    out_encoder_flat = Flatten()(out_encoder)
    input_memory = Concatenate()([out_encoder_flat, input_prev_action, input_prev_reward])
    out_memory, out_memory_states = l_memory(input_memory)

    ### backbone
    if shared_backbone:
        out_backbone = l_backbone_shared(out_memory)
        out_backbone_a = out_backbone
        out_backbone_c = out_backbone
    else:
        out_backbone_a = l_backbone_a(out_memory)
        out_backbone_c = l_backbone_c(out_memory)

    ### heads
    out_actor = l_actor_head(out_backbone_a)
    out_critic = l_critic_head(out_backbone_c)

    #

    Actor = Model(
        inputs=[input_obs, input_prev_action, input_prev_reward],
        outputs=out_actor,
    )

    Critic = Model(
        inputs=[input_obs, input_prev_action, input_prev_reward],
        outputs=out_critic,
    )

    MetaMemory = Model(
        inputs=[input_obs, input_prev_action, input_prev_reward],
        outputs=out_memory_states,
    )

    return Actor, Critic, MetaMemory
