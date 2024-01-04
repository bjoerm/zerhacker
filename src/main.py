import shutil
import tomllib

from initiator import Initiator
from pydantic_config import Config

with open("src/config.toml", "rb") as f:
    config = tomllib.load(f)

config = Config(**config)

files_for_splitter = Initiator(input_folder=config.paths.path_untouched_scans, output_folder=config.paths.path_rough_cut).init()


# TODO Do splitter work here.

files_for_fine_cut = Initiator(input_folder=config.paths.path_rough_cut, output_folder=config.paths.path_fine_cut).init()
