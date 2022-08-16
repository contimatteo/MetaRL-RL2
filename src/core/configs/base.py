from typing import List

###

MODES_ENUM = ["training", "inference", "render"]
AGENTS_ENUM = ["AC", "A2C", "A3C", "MetaA2C", "MetaA3C"]

###


class ConfigScheme():

    def __init__(self) -> None:
        self._raw: dict = None
        self.mode: str = None
        self.n_trials: int = None
        self.n_episodes: int = None
        self.n_explore_episodes: int = None
        self.n_max_steps: int = None
        self.envs: List[dict] = None
        self.agent: str = None
        self.policy: str = None

    #

    def _validate_raw_config(self) -> None:
        assert self._raw is not None
        assert isinstance(self._raw, dict)

        assert "mode" in self._raw
        mode = self._raw["mode"]
        assert isinstance(mode, str)
        assert mode in MODES_ENUM

        assert "n_trials" in self._raw
        n_trials = self._raw["n_trials"]
        assert isinstance(n_trials, int) and n_trials > 0
        assert "n_episodes" in self._raw
        n_episodes = self._raw["n_episodes"]
        assert isinstance(n_episodes, int) and n_episodes > 0
        assert "n_max_steps" in self._raw
        n_max_steps = self._raw["n_max_steps"]
        assert isinstance(n_max_steps, int) and n_max_steps > 0
        assert "n_explore_episodes" in self._raw
        n_explore_episodes = self._raw["n_explore_episodes"]
        assert isinstance(n_explore_episodes, int) and n_explore_episodes > 0

        assert "envs" in self._raw
        envs = self._raw["envs"]
        assert isinstance(envs, list) and len(envs) > 0
        for env in envs:
            assert isinstance(env, dict)
            assert isinstance(env["name"], str)
            assert isinstance(env["params"], dict)

    def _parse_raw_config(self) -> None:
        self.mode = self._raw["mode"]
        self.n_trials = self._raw["n_trials"]
        self.n_episodes = self._raw["n_episodes"]
        self.n_explore_episodes = self._raw["n_explore_episodes"]
        self.n_max_steps = self._raw["n_max_steps"]
        self.envs = self._raw["envs"]
        self.agent = self._raw["agent"]
        self.policy = self._raw["policy"]

    def _validate_parsed_config(self) -> None:
        assert self._raw is not None
        assert self.mode is not None
        assert self.n_trials is not None
        assert self.n_episodes is not None
        assert self.n_explore_episodes is not None
        assert self.n_max_steps is not None
        assert self.agent is not None
        assert self.policy is not None
        assert isinstance(self.envs, list) and len(self.envs) > 0

    #

    def initialize(self, config: dict) -> None:
        self._raw = config
