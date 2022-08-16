from typing import List, Union, Optional
from pathlib import Path

import json
import gym

from gym.spaces import Discrete
from gym.spaces import Box
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import load_model

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
        self.critic_nn_opt = None

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
    def meta_policy(self) -> bool:
        if self.agent is not None:
            assert self.agent.meta_algorithm == self._config.meta_policy
            return self._config.meta_policy

        return self._config.meta_policy

    @property
    def env_actions_bounds(self) -> Optional[list]:
        if self.env_actions_are_discrete:
            return None

        action_space = self.env_action_space
        return [action_space.low, action_space.high]

    @property
    def actor_nn_weights_url(self) -> Path:
        weights_url = LocalStorageManager.dirs.tmp_saved_models
        weights_url = weights_url.joinpath(self._config.trial_id)
        weights_url = weights_url.joinpath("actor")
        if not weights_url.is_dir():
            weights_url.mkdir(exist_ok=True, parents=True)
        return weights_url

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

    #

    def _load_envs(self) -> None:
        self.envs = []

        for env in self._config.envs:
            env = ControllerUtils.gym_env(env["name"], env["params"])
            self.envs.append(env)

    def _load_networks(self) -> None:
        obs_space = self.env_obs_space
        action_space = self.env_action_space
        meta_network = self.meta_policy
        shared_backbone = self._config.nn_shared_backbone

        self.actor_nn, self.critic_nn, self.memory_nn = ControllerUtils.networks(
            obs_space=obs_space,
            action_space=action_space,
            meta_network=meta_network,
            shared_backbone=shared_backbone
        )

        self.actor_nn_opt = ControllerUtils.network_optimizer("RMSProp")
        self.critic_nn_opt = ControllerUtils.network_optimizer("RMSProp")

    def _load_policy(self) -> None:
        policy_name = self._config.policy
        obs_space = self.env_obs_space
        action_space = self.env_action_space
        action_bounds = self.env_actions_bounds

        assert self.actor_nn is not None

        if policy_name == "Random":
            if self.meta_policy:
                self.policy = RandomMetaPolicy(
                    state_space=obs_space, action_space=action_space, action_buonds=action_bounds
                )
            else:
                self.policy = RandomPolicy(
                    state_space=obs_space, action_space=action_space, action_buonds=action_bounds
                )
            return

        if policy_name == "Network":
            if self.meta_policy:
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
            if self.meta_policy:
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

    def _load_agent(self) -> None:
        agent_name = self._config.agent
        n_max_steps = self._config.n_max_steps

        if self.meta_policy:
            assert self.memory_nn is not None

        if agent_name == "A2C":
            if self.meta_policy:
                self.agent = MetaA2C(
                    n_max_episode_steps=n_max_steps,
                    policy=self.policy,
                    actor_network=self.actor_nn,
                    critic_network=self.critic_nn,
                    actor_network_opt=self.actor_nn_opt,
                    critic_network_opt=self.critic_nn_opt,
                    memory_network=self.memory_nn,
                )
            else:
                self.agent = A2C(
                    n_max_episode_steps=n_max_steps,
                    policy=self.policy,
                    actor_network=self.actor_nn,
                    critic_network=self.critic_nn,
                    actor_network_opt=self.actor_nn_opt,
                    critic_network_opt=self.critic_nn_opt,
                )
            return

        if agent_name == "A3C":
            if self.meta_policy:
                self.agent = MetaA3C(
                    n_max_episode_steps=n_max_steps,
                    policy=self.policy,
                    actor_network=self.actor_nn,
                    critic_network=self.critic_nn,
                    actor_network_opt=self.actor_nn_opt,
                    critic_network_opt=self.critic_nn_opt,
                    memory_network=self.memory_nn,
                )
            else:
                self.agent = A3C(
                    n_max_episode_steps=n_max_steps,
                    policy=self.policy,
                    actor_network=self.actor_nn,
                    critic_network=self.critic_nn,
                    actor_network_opt=self.actor_nn_opt,
                    critic_network_opt=self.critic_nn_opt,
                )
            return

        raise Exception("agent not supported.")

    #

    def _load_trained_models(self) -> None:
        weights_url = self.actor_nn_weights_url

        assert weights_url.exists()
        assert weights_url.is_dir()
        assert not LocalStorageManager.is_empty(weights_url)

        self.actor_nn = load_model(weights_url)

    def _save_trained_models(self) -> None:
        weights_url = self.actor_nn_weights_url

        self.actor_nn.save(weights_url, overwrite=True)

    def _validate_controller(self) -> None:
        envs = self.envs

        for i in range(len(envs) - 1):
            assert isinstance(envs[i], gym.Env)
            assert envs[i].observation_space == envs[i + 1].observation_space
            assert envs[i].action_space == envs[i + 1].action_space

        assert self.policy is not None

        if self.mode == "training":
            assert self.actor_nn is not None
            assert self.critic_nn is not None
            assert self.actor_nn_opt is not None
            assert self.critic_nn_opt is not None
            assert self.agent is not None

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

        # self.__load_envs()
        # self.__load_networks()
        # self.__load_policy()
        # self.__load_agent()
        # self.__validate_controller()
        # self._initialized = True

        return
