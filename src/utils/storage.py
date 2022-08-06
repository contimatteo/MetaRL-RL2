# import os

# from pathlib import Path

# ###

# class LocalStorageDirectoryManager:

#     def __init__(self) -> None:
#         self.root = None
#         self.tmp = None
#         self.tmp_datasets = None
#         self.tmp_nltk = None
#         self.tmp_checkpoints = None
#         self.tmp_saved_models = None

#         self.__initialize()
#         self.__mkdirs()
#         self.__validate()

#     #

#     def __initialize(self) -> None:
#         self.root = Path(__file__).parent.parent.parent

#         self.tmp = self.root.joinpath('tmp')
#         self.tmp_datasets = self.tmp.joinpath('datasets')
#         self.tmp_nltk = self.tmp.joinpath('nltk')
#         self.tmp_checkpoints = self.tmp.joinpath('checkpoints')
#         self.tmp_saved_models = self.tmp.joinpath('saved_models')

#     def __mkdirs(self) -> None:
#         self.tmp.mkdir(exist_ok=True)
#         self.tmp_datasets.mkdir(exist_ok=True)
#         self.tmp_nltk.mkdir(exist_ok=True)
#         self.tmp_checkpoints.mkdir(exist_ok=True)
#         self.tmp_saved_models.mkdir(exist_ok=True)

#     def __validate(self) -> None:
#         assert self.root.is_dir()

#         assert self.tmp.is_dir()
#         assert self.tmp_datasets.is_dir()
#         assert self.tmp_nltk.is_dir()
#         assert self.tmp_checkpoints.is_dir()
#         assert self.tmp_saved_models.is_dir()

# ###

# class LocalStorageManager():

#     def __init__(self) -> None:
#         self.dirs = LocalStorageDirectoryManager()

#     #

#     @staticmethod
#     def is_empty(directory: Path) -> bool:
#         # return any(Path(directory).iterdir())
#         return len(os.listdir(str(directory))) < 1

#     def nn_checkpoint_url(self, model_name: str) -> Path:
#         return self.dirs.tmp_checkpoints.joinpath(f"{model_name}.h5")

#     def nn_saved_model_url(self, model_name: str) -> Path:
#         return self.dirs.tmp_saved_models.joinpath(f"{model_name}")

# ###

# # class StoreObject():
# #     def __init__(self, object_name: str) -> None:
# #         super().__init__()
# #         self.object_path = os.path.join(self.tmp, object_name)
# #     def load_object(self):
# #         if not os.path.exists(self.object_path):
# #             return None
# #         with open(self.object_path, "rb") as f:
# #             return pickle.load(f)
# #     def save_object(self, object_to_save):
# #         with open(self.object_path, "wb") as f:
# #             pickle.dump(object_to_save, f)
# #     def exists(self):
# #         return os.path.exists(self.object_path)
