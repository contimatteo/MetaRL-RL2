import os

from pathlib import Path

###


class LocalStorageDirectoryManagerClass:

    def __init__(self) -> None:
        self.root = None

        self.configs = None

        self.tmp = None
        self.tmp_saved_models = None
        self.tmp_history = None

        self.__initialize()
        self.__mkdirs()
        self.__validate()

    #

    def __initialize(self) -> None:
        self.root = Path(__file__).parent.parent.parent

        self.configs = self.root.joinpath("configs")

        self.tmp = self.root.joinpath('tmp')
        self.tmp_saved_models = self.tmp.joinpath('saved_models')
        self.tmp_history = self.tmp.joinpath('history')

    def __mkdirs(self) -> None:
        self.configs.mkdir(exist_ok=True)
        self.tmp.mkdir(exist_ok=True)
        self.tmp_saved_models.mkdir(exist_ok=True)
        self.tmp_history.mkdir(exist_ok=True)

    def __validate(self) -> None:
        assert self.root.is_dir()
        assert self.configs.is_dir()
        assert self.tmp.is_dir()
        assert self.tmp_saved_models.is_dir()
        assert self.tmp_history.is_dir()


###


class LocalStorageManagerClass():

    def __init__(self) -> None:
        self.dirs = LocalStorageDirectoryManagerClass()

    #

    @staticmethod
    def is_empty(directory: Path) -> bool:
        # return any(Path(directory).iterdir())
        return len(os.listdir(str(directory))) < 1


###

LocalStorageManager = LocalStorageManagerClass()
