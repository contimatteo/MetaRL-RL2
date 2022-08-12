from typing import Tuple

import gym

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Flatten, Concatenate, Reshape

from networks.layers import A_HeadLayer, C_HeadLayer
from networks.layers import AC_EncoderLayer, AC_BackboneLayer, AC_MetaMemoryLayer

###


def MetaActorCriticNetworks(obs_space: gym.Space, action_space: gym.Space,
                            batch_size: int) -> Tuple[Model, Model]:
    ### TODO: support also 'continuous' action space
    assert isinstance(action_space, gym.spaces.discrete.Discrete)
    discrete = True

    obs_shape = obs_space.shape

    ### input
    input_obs = Input(shape=obs_shape, batch_size=batch_size, name="Input_Observations")
    input_prev_action = Input(shape=(1, ), name="Input_PreviousAction")
    input_prev_reward = Input(shape=(1, ), name="Input_PreviousReward")
    ### encoder
    l_encoder = AC_EncoderLayer()
    ### meta-memory
    l_memory = AC_MetaMemoryLayer(name="MetaMemoryLayer")
    ### backbone
    l_shared_backbone = AC_BackboneLayer()
    ### head
    l_actor_head = A_HeadLayer(action_space.n, discrete=discrete)
    l_critic_head = C_HeadLayer()

    #

    ### encoder
    out_encoder = l_encoder(input_obs)

    ### memory
    out_encoder_flat = Flatten()(out_encoder)
    input_memory = Concatenate()([out_encoder_flat, input_prev_action, input_prev_reward])
    out_memory, _ = l_memory(input_memory)

    ### backbone
    out_backbone = l_shared_backbone(out_memory)

    ### heads
    out_actor = l_actor_head(out_backbone)
    out_critic = l_critic_head(out_backbone)

    #

    Actor = Model(
        inputs=[input_obs, input_prev_action, input_prev_reward],
        outputs=out_actor,
    )

    Critic = Model(
        inputs=[input_obs, input_prev_action, input_prev_reward],
        outputs=out_critic,
    )

    return Actor, Critic
