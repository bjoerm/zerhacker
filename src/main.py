import tomllib

from initiator import Initiator
from pydantic_config import Config
from scanned_album_page import ScannedAlbumPage

with open("src/config.toml", "rb") as f:
    config = tomllib.load(f)

config = Config.model_validate(config)

files_for_splitter = Initiator(input_folder=config.paths.path_untouched_scans, output_folder=config.paths.path_splitter).init()

for file in files_for_splitter:
    ScannedAlbumPage(
        img_path_input=file, folder_input=config.paths.path_untouched_scans, folder_output=config.paths.path_splitter, min_pixel_ratio=config.splitter.min_pixel_ratio, debug_mode=True
    ).split_scanned_image()


# files_for_fine_cut = Initiator(input_folder=config.paths.path_splitter, output_folder=config.paths.path_fine_cut, debug_mode=True).init()
