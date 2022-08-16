from .inference import InferenceConfig

###


class TrainingConfig(InferenceConfig):

    def __init__(self) -> None:
        super().__init__()

        self.n_explore_episodes: int = None
        self.batch_size = None
        self.nn_shared_backbone = None
        self.actor_opt_lr = None
        self.critic_opt_lr = None
        self.agent: str = None

    #

    def _validate_raw_config(self) -> None:
        super()._validate_raw_config()

        assert "n_explore_episodes" in self._raw
        n_explore_episodes = self._raw["n_explore_episodes"]
        assert isinstance(n_explore_episodes, int) and n_explore_episodes > 0

        assert "batch_size" in self._raw
        batch_size = self._raw["batch_size"]
        assert isinstance(batch_size, int) and batch_size > 0

        assert "nn_shared_backbone" in self._raw
        nn_shared_backbone = self._raw["nn_shared_backbone"]
        assert isinstance(nn_shared_backbone, bool)

        assert "actor_opt_lr" in self._raw
        actor_opt_lr = self._raw["actor_opt_lr"]
        assert isinstance(actor_opt_lr, float)
        assert "critic_opt_lr" in self._raw
        critic_opt_lr = self._raw["critic_opt_lr"]
        assert isinstance(critic_opt_lr, float)

        assert "agent" in self._raw
        agent = self._raw["agent"]
        assert isinstance(agent, str)

    def _parse_raw_config(self) -> None:
        super()._parse_raw_config()

        self.n_explore_episodes = self._raw["n_explore_episodes"]
        self.batch_size = self._raw["batch_size"]
        self.nn_shared_backbone = self._raw["nn_shared_backbone"]
        self.actor_opt_lr = self._raw["actor_opt_lr"]
        self.critic_opt_lr = self._raw["critic_opt_lr"]
        self.agent = self._raw["agent"]

    def _validate_parsed_config(self) -> None:
        super()._validate_parsed_config()

        assert self.n_explore_episodes is not None
        assert self.batch_size is not None
        assert self.nn_shared_backbone is not None
        assert self.actor_opt_lr is not None
        assert self.critic_opt_lr is not None
        assert self.agent is not None

    #

    def initialize(self, config: dict) -> None:
        super().initialize(config)

        self._validate_raw_config()
        self._parse_raw_config()
        self._validate_parsed_config()
