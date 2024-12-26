# TODO Check and re-enable _filter_out_too_small_contours() and _filter_out_contours_with_odd_width_height_ratios()
# TODO Should cv2.Canny be used? https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
# TODO Add min_pixel_ratio in extracted_image class as base check.

from fine_cut.extracted_image import ExtractedImage
from shared.initiator import Initiator
from shared.logger import logger
from shared.pydantic_config import load_config

config = load_config()

logger.info("Only fine cut started")

files_for_fine_cut = Initiator(input_folder=config.paths.path_splitter, output_folder=config.paths.path_fine_cut).init()

for file in files_for_fine_cut:
    ExtractedImage(
        image_path_input=file,
        folder_input=config.paths.path_splitter,
        folder_output=config.paths.path_fine_cut,
        manual_detection_threshold=config.fine_cut.manual_detection_threshold,
        min_pixel_ratio=config.fine_cut.min_pixel_ratio,
        debug_mode=True,
        write_mode=True,
    ).rotate_and_crop(extra_crop=config.fine_cut.extra_crop)

logger.info("Only fine cut started finished")
