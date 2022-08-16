from .base import ConfigScheme

###


class InferenceConfig(ConfigScheme):

    def __init__(self) -> None:
        super().__init__()

        self.meta_policy: bool = None

    #

    def _validate_raw_config(self) -> None:
        super()._validate_raw_config()

        assert "meta_policy" in self._raw
        meta_policy = self._raw["meta_policy"]
        assert isinstance(meta_policy, bool)

    def _parse_raw_config(self) -> None:
        super()._parse_raw_config()

        self.meta_policy = self._raw["meta_policy"]

    def _validate_parsed_config(self) -> None:
        super()._validate_parsed_config()

        assert self.meta_policy is not None

    #

    def initialize(self, config: dict) -> None:
        super().initialize(config)

        self._validate_raw_config()
        self._parse_raw_config()
        self._validate_parsed_config()
