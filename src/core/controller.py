from typing import List, Union, Optional
from pathlib import Path

import json
import gym

from gym.spaces import Discrete
from gym.spaces import Box
from tensorflow.python.keras import Model

from agents import AC
from agents import A2C
from agents import A3C
from agents import MetaA2C
from agents import MetaA3C
from core.configs import TrainingConfig
from core.configs import InferenceConfig
from core.configs import RenderConfig
from policies import Policy
from policies import MetaPolicy
from policies import RandomPolicy
from policies import RandomMetaPolicy
from policies import NetworkPolicy
from policies import NetworkMetaPolicy
from policies import EpsilonGreedyPolicy
from utils import ControllerUtils
from utils import LocalStorageManager

###


class Controller():

    def __init__(self) -> None:
        self._initialized = False

        self._raw_config = None
        self._config: Union[TrainingConfig, Union[InferenceConfig, RenderConfig]] = None

        self.actor_nn: Model = None
        self.critic_nn: Model = None
        self.memory_nn: Optional[Model] = None

        self.actor_nn_opt = None
        self.crtici_nn_opt = None

        self.envs: List[gym.Env] = []
        self.policy: Union[Policy, MetaPolicy] = None
        self.agent: AC = None

    #

    @property
    def mode(self) -> str:
        return self._config.mode

    @property
    def env_action_space(self) -> Union[Discrete, Box]:
        return self.envs[0].action_space

    @property
    def env_obs_space(self) -> Box:
        return self.envs[0].observation_space

    @property
    def env_actions_are_discrete(self) -> bool:
        return isinstance(self.env_action_space, Discrete)

    @property
    def meta_agent(self) -> bool:
        if self.agent is not None:
            assert self.agent.meta_algorithm == self._config.meta_agent
            return self._config.meta_agent

        return self._config.meta_agent

    def env_actions_bounds(self) -> Optional[list]:
        if self.env_actions_are_discrete:
            return None

        action_space = self.env_action_space
        return [action_space.low, action_space.high]

    #

    def __validate_raw_config(self) -> None:
        assert self._raw_config is not None
        assert isinstance(self._raw_config, dict)

    def __parse_raw_config(self) -> None:
        if self._raw_config["mode"] == "training":
            self._config = TrainingConfig()
            self._config.initialize(self._raw_config)
            return

        if self._raw_config["mode"] == "inference":
            self._config = InferenceConfig()
            self._config.initialize(self._raw_config)
            return

        if self._raw_config["mode"] == "render":
            self._config = RenderConfig()
            self._config.initialize(self._raw_config)
            return

        raise Exception("configuration mode not supported.")

    def __validate_parsed_config(self) -> None:
        assert self._config is not None
        assert isinstance(self._config, TrainingConfig) or isinstance(
            self._config, InferenceConfig
        ) or isinstance(self._config, RenderConfig)

    def __validate_controller(self) -> None:
        envs = self.envs

        for i in range(len(envs) - 1):
            assert isinstance(envs[i], gym.Env)
            assert envs[i].observation_space == envs[i + 1].observation_space
            assert envs[i].action_space == envs[i + 1].action_space

        assert self.actor_nn is not None
        assert self.critic_nn is not None
        assert self.actor_nn_opt is not None
        assert self.crtici_nn_opt is not None
        assert self.policy is not None
        assert self.agent is not None

    #

    def __load_envs(self) -> None:
        self.envs = []

        for env in self._config.envs:
            env = ControllerUtils.gym_env(env["name"], env["params"])
            self.envs.append(env)

    def __load_networks(self) -> None:
        obs_space = self.env_obs_space
        action_space = self.env_action_space
        meta_network = self.meta_agent
        shared_backbone = self._config.nn_shared_backbone

        self.actor_nn, self.critic_nn, self.memory_nn = ControllerUtils.networks(
            obs_space=obs_space,
            action_space=action_space,
            meta_network=meta_network,
            shared_backbone=shared_backbone
        )

        self.actor_nn_opt = ControllerUtils.network_optimizer("RMSProp")
        self.crtici_nn_opt = ControllerUtils.network_optimizer("RMSProp")

    def __load_policy(self) -> None:
        policy_name = self._config.policy
        obs_space = self.env_obs_space
        action_space = self.env_action_space
        action_bounds = self.env_actions_bounds

        assert self.actor_nn is not None

        if policy_name == "Random":
            if self.meta_agent:
                self.policy = RandomMetaPolicy(
                    state_space=obs_space, action_space=action_space, action_buonds=action_bounds
                )
            else:
                self.policy = RandomPolicy(
                    state_space=obs_space, action_space=action_space, action_buonds=action_bounds
                )
            return

        if policy_name == "Network":
            if self.meta_agent:
                self.policy = NetworkMetaPolicy(
                    state_space=obs_space,
                    action_space=action_space,
                    network=self.actor_nn,
                    action_buonds=action_bounds
                )
            else:
                self.policy = NetworkPolicy(
                    state_space=obs_space,
                    action_space=action_space,
                    network=self.actor_nn,
                    action_buonds=action_bounds
                )
            return

        if policy_name == "EpsilonGreedy":
            nn_policy = None
            if self.meta_agent:
                nn_policy = NetworkMetaPolicy(
                    state_space=obs_space,
                    action_space=action_space,
                    network=self.actor_nn,
                    action_buonds=action_bounds
                )
            else:
                nn_policy = NetworkPolicy(
                    state_space=obs_space,
                    action_space=action_space,
                    network=self.actor_nn,
                    action_buonds=action_bounds
                )
            self.policy = EpsilonGreedyPolicy(policy=nn_policy, action_buonds=action_bounds)
            return

        raise Exception("policy not supported")

    def __load_agent(self) -> None:
        agent_name = self._config.agent
        n_max_steps = self._config.n_max_steps

        if self.meta_agent:
            assert self.memory_nn is not None

        if agent_name == "A2C":
            if self.meta_agent:
                self.agent = MetaA2C(
                    n_max_episode_steps=n_max_steps,
                    policy=self.policy,
                    actor_network=self.actor_nn,
                    critic_network=self.critic_nn,
                    actor_network_opt=self.actor_nn_opt,
                    critic_network_opt=self.crtici_nn_opt,
                    memory_network=self.memory_nn,
                )
            else:
                self.agent = A2C(
                    n_max_episode_steps=n_max_steps,
                    policy=self.policy,
                    actor_network=self.actor_nn,
                    critic_network=self.critic_nn,
                    actor_network_opt=self.actor_nn_opt,
                    critic_network_opt=self.crtici_nn_opt,
                )
            return

        if agent_name == "A3C":
            if self.meta_agent:
                self.agent = MetaA3C(
                    n_max_episode_steps=n_max_steps,
                    policy=self.policy,
                    actor_network=self.actor_nn,
                    critic_network=self.critic_nn,
                    actor_network_opt=self.actor_nn_opt,
                    critic_network_opt=self.crtici_nn_opt,
                    memory_network=self.memory_nn,
                )
            else:
                self.agent = A3C(
                    n_max_episode_steps=n_max_steps,
                    policy=self.policy,
                    actor_network=self.actor_nn,
                    critic_network=self.critic_nn,
                    actor_network_opt=self.actor_nn_opt,
                    critic_network_opt=self.crtici_nn_opt,
                )
            return

        raise Exception("agent not supported.")

    #

    def intialize(self, config_file_url: Path) -> None:
        assert isinstance(config_file_url, Path)

        # config_file = Path(config_file_url)
        assert config_file_url.exists()
        assert config_file_url.is_file()

        with open(str(config_file_url), 'r', encoding="utf-8") as file:
            self._raw_config = json.load(file)

        assert isinstance(self._raw_config, dict)
        assert len(self._raw_config.keys()) > 0

        self.__validate_raw_config()
        self.__parse_raw_config()
        self.__validate_parsed_config()

        self.__load_envs()
        self.__load_networks()
        self.__load_policy()
        self.__load_agent()

        self.__validate_controller()

        self._initialized = True

    #

    def run(self) -> None:
        assert self._initialized is True
