from typing import Tuple, Optional

import gym

import metagym.quadrotor
import metagym.quadrupedal

from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizers import Optimizer
from tensorflow.python.keras.optimizers import rmsprop_v2

from environments import BanditTwoArmedDependentEasy
from environments import BanditTwoArmedDependentMedium
from environments import BanditTwoArmedDependentHard
from environments import BanditTenArmedRandomRandom
from environments import BanditTenArmedRandomFixed
from networks import ActorCriticNetworks
from networks import MetaActorCriticNetworks

###


class ControllerUtils():

    @staticmethod
    def gym_env(name: str, params: dict, render: bool) -> gym.Env:
        assert isinstance(params, dict)

        if name == "gym/Ant":
            return gym.make("Ant-v4")

        if name == "gym/CartPole":
            return gym.make("CartPole-v0")

        if name == "gym/BipedalWalker":
            return gym.make("BipedalWalker-v3")

        if name == "gym/LunarLander":
            enable_wind = params["enable_wind"] if "enable_wind" in params else None
            wind_power = params["wind_power"] if "wind_power" in params else None

            assert isinstance(enable_wind, bool)
            assert isinstance(wind_power, float)

            return gym.make("LunarLander-v2", continuous=True)

        if name == "bandits/TwoArmedDependentEasy":
            return BanditTwoArmedDependentEasy()

        if name == "bandits/TwoArmedDependentMedium":
            return BanditTwoArmedDependentMedium()

        if name == "bandits/TwoArmedDependentHard":
            return BanditTwoArmedDependentHard()

        if name == "bandits/TenArmedRandomRandom":
            return BanditTenArmedRandomRandom()

        if name == "bandits/TenArmedRandomFixed":
            return BanditTenArmedRandomFixed()

        if name == "metagym/Quadrotor":
            task = params["task"] if "task" in params else None

            assert isinstance(task, str)
            assert task in ["no_collision", "velocity_control", "hovering_control"]

            return gym.make("quadrotor-v0", task=task)

        if name == "metagym/Quadrupedal":
            task = params["task"] if "task" in params else None

            assert isinstance(task, str)
            assert task in [
                "stairstair",
                "stairslope",
                "slopestair",
                "slopeslope",
                "stair13",
                "terrain",
                "balancebeam",
                "gallop",
                "Cave",
                "ground",
            ]

            if render:
                return gym.make('quadrupedal-v0', render=1, task=task)
            else:
                return gym.make('quadrupedal-v0', task=task)

        #

        raise Exception("environment name not supported.")

    @staticmethod
    def networks(
        obs_space: gym.Space,
        action_space: gym.Space,
        meta_network: bool,
        shared_backbone: bool = False
    ) -> Tuple[Model, Model, Optional[Model]]:
        actor_nn = None
        critic_nn = None
        memory_nn = None

        if meta_network:
            actor_nn, critic_nn, memory_nn = MetaActorCriticNetworks(
                obs_space, action_space, shared_backbone=shared_backbone
            )
        else:
            actor_nn, critic_nn = ActorCriticNetworks(
                obs_space, action_space, shared_backbone=shared_backbone
            )

        return actor_nn, critic_nn, memory_nn

    @staticmethod
    def network_optimizer(name: str, learning_rate: float = 1e-4) -> Optimizer:
        if name == "RMSProp":
            return rmsprop_v2.RMSProp(learning_rate=learning_rate)

        raise Exception("optimizer not supported.")
