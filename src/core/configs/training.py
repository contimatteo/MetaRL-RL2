from .base import ConfigScheme

###


class TrainingConfig(ConfigScheme):

    def __init__(self) -> None:
        super().__init__()

        self.batch_size = None

        self.meta_agent = None
        self.nn_shared_backbone = None

        self.actor_opt_lr = None
        self.critic_opt_lr = None

    #

    def _validate_raw_config(self) -> None:
        super()._validate_raw_config()

        assert "batch_size" in self._raw
        batch_size = self._raw["batch_size"]
        assert isinstance(batch_size, int) and batch_size > 0

        assert "meta_agent" in self._raw
        meta_agent = self._raw["meta_agent"]
        assert isinstance(meta_agent, bool)
        assert "nn_shared_backbone" in self._raw
        nn_shared_backbone = self._raw["nn_shared_backbone"]
        assert isinstance(nn_shared_backbone, bool)

        assert "actor_opt_lr" in self._raw
        actor_opt_lr = self._raw["actor_opt_lr"]
        assert isinstance(actor_opt_lr, float)
        assert "critic_opt_lr" in self._raw
        critic_opt_lr = self._raw["critic_opt_lr"]
        assert isinstance(critic_opt_lr, float)

    def _parse_raw_config(self) -> None:
        super()._parse_raw_config()

        self.batch_size = self._raw["batch_size"]
        self.meta_agent = self._raw["meta_agent"]
        self.nn_shared_backbone = self._raw["nn_shared_backbone"]
        self.actor_opt_lr = self._raw["actor_opt_lr"]
        self.critic_opt_lr = self._raw["critic_opt_lr"]

    def _validate_parsed_config(self) -> None:
        super()._validate_parsed_config()

        assert self.batch_size is not None
        assert self.meta_agent is not None
        assert self.nn_shared_backbone is not None
        assert self.actor_opt_lr is not None
        assert self.critic_opt_lr is not None

    #

    def initialize(self, config: dict) -> None:
        super().initialize(config)

        self._validate_raw_config()
        self._parse_raw_config()
        self._validate_parsed_config()
