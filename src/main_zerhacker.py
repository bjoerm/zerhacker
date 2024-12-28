# TODO Should cv2.Canny be used? https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html

from multiprocessing import Pool
from typing import Optional

import tqdm

from fine_cut.extracted_image import ExtractedImage
from shared.initiator import Initiator
from shared.logger import logger
from shared.multi_threading import get_available_threads
from shared.pydantic_args import Args
from shared.pydantic_config import load_config
from splitter.scanned_album_page import ScannedAlbumPage


def zerhacker_parallel():
    """Parallel execution of the Zerhacker."""

    logger.info("Zerhacker started")
    config = load_config()

    config.parallel.num_threads = get_available_threads()

    files_for_splitter = Initiator(input_folder=config.paths.path_untouched_scans, output_folder=config.paths.path_splitter).init()

    args = []

    for file in files_for_splitter:
        arg = None
        arg = Args(
            task="split",
            image_path_input=file,
            folder_input=config.paths.path_untouched_scans,
            folder_output=config.paths.path_splitter,
            manual_detection_threshold=config.splitter.manual_detection_threshold,
            min_pixel_ratio=config.splitter.min_pixel_ratio,
            debug_mode=config.general.debug_mode,
            write_mode=config.general.write_mode,
            extra_crop=config.fine_cut.extra_crop,
        )

        args.append(arg)

    parallel_executor(num_threads=config.parallel.num_threads, args=args)

    files_for_fine_cut = Initiator(input_folder=config.paths.path_splitter, output_folder=config.paths.path_fine_cut).init()

    args = []

    for file in files_for_fine_cut:
        arg = None
        arg = Args(
            task="fine_cut",
            image_path_input=file,
            folder_input=config.paths.path_splitter,
            folder_output=config.paths.path_fine_cut,
            manual_detection_threshold=config.fine_cut.manual_detection_threshold,
            min_pixel_ratio=config.fine_cut.min_pixel_ratio,
            debug_mode=config.general.debug_mode,
            write_mode=config.general.write_mode,
            extra_crop=config.fine_cut.extra_crop,
        )

        args.append(arg)

    parallel_executor(num_threads=config.parallel.num_threads, args=args)


def _parallel_class_init(args: Args):
    """Function is called in parallel."""

    if args.task == "split":
        split = ScannedAlbumPage(args.image_path_input, args.folder_input, args.folder_output, args.manual_detection_threshold, args.min_pixel_ratio, args.debug_mode, args.write_mode)
        split.split_scanned_image()

    elif args.task == "fine_cut":
        fine_cut = ExtractedImage(args.image_path_input, args.folder_input, args.folder_output, args.manual_detection_threshold, args.min_pixel_ratio, args.debug_mode, args.write_mode)
        fine_cut.rotate_and_crop(extra_crop=args.extra_crop)

    else:
        raise ValueError("Undefined task.")


def parallel_executor(num_threads: Optional[int], args: list[Args]):
    """Function to call the function in parallel."""

    logger.info(f"{args[0].task} parallelization started")

    with Pool(num_threads) as p:
        list(tqdm.tqdm(p.imap(_parallel_class_init, args), total=len(args)))

    logger.info(f"{args[0].task} parallelization finished")


if __name__ == "__main__":
    zerhacker_parallel()
