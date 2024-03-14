# TODO Fix detection of rotated pictures as well as images with sky.


from fine_cut.extracted_image import ExtractedImage
from shared.initiator import Initiator
from shared.pydantic_config import load_config
from splitter.scanned_album_page import ScannedAlbumPage

config = load_config()

files_for_splitter = Initiator(input_folder=config.paths.path_untouched_scans, output_folder=config.paths.path_splitter).init()

for file in files_for_splitter:
    ScannedAlbumPage(
        image_path_input=file,
        folder_input=config.paths.path_untouched_scans,
        folder_output=config.paths.path_splitter,
        manual_threshold=config.splitter.manual_detection_threshold,
        min_pixel_ratio=config.splitter.min_pixel_ratio,
        debug_mode=True,
        write_mode=True,
    ).split_scanned_image()


files_for_fine_cut = Initiator(input_folder=config.paths.path_splitter, output_folder=config.paths.path_fine_cut).init()

for file in files_for_fine_cut:
    ExtractedImage(
        image_path_input=file,
        folder_input=config.paths.path_splitter,
        folder_output=config.paths.path_fine_cut,
        manual_threshold=config.fine_cut.manual_detection_threshold,
        min_pixel_ratio=config.fine_cut.min_pixel_ratio,
        debug_mode=True,
        write_mode=True,
    ).rotate_and_crop(extra_crop=config.fine_cut.extra_crop)
