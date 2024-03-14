import shutil
from pathlib import Path

import pytest

from shared.pydantic_config import load_config
from splitter.scanned_album_page import ScannedAlbumPage


@pytest.fixture
def write_mode():
    """Central switch for writing files into the path_folder_temp_output folder."""
    return True


@pytest.fixture
def path_folder_input():
    return Path("tests/splitter/input")


@pytest.fixture
def path_folder_temp_output():
    return Path("tests/splitter/temp_output")


@pytest.fixture
def config():
    return load_config()


def test_remove_temp_output_folder(path_folder_temp_output):
    """Tidying up before launching new tests."""
    shutil.rmtree(path_folder_temp_output, ignore_errors=True)


@pytest.mark.parametrize(
    "input_path, expected_output",
    [
        ["tests/splitter/input/0_only_back.jpg", 0],
        ["tests/splitter/input/0_white_background.jpg", 0],
        ["tests/splitter/input/1a_three_images_overlap.jpg", 1],
        ["tests/splitter/input/1b_five_images_overlap.jpg", 1],
        ["tests/splitter/input/2a.jpg", 2],
        ["tests/splitter/input/2b_one_more_only_back.jpg", 2],
        ["tests/splitter/input/2c_diag.jpg", 2],
        ["tests/splitter/input/2d.jpg", 2],
        ["tests/splitter/input/2e_two_more_with_strong_reflection.jpg", 2],
        ["tests/splitter/input/3a.jpg", 3],
        ["tests/splitter/input/3b.jpg", 3],
        ["tests/splitter/input/4a.jpg", 4],
        ["tests/splitter/input/4b.jpg", 4],
    ],
)
def test_found_contours(config, path_folder_input, path_folder_temp_output, write_mode, input_path, expected_output):

    album_page = ScannedAlbumPage(
        image_path_input=Path(input_path),
        folder_input=path_folder_input,
        folder_output=path_folder_temp_output,
        manual_threshold=config.splitter.manual_detection_threshold,
        min_pixel_ratio=config.splitter.min_pixel_ratio,
        debug_mode=True,
        write_mode=write_mode,
    )

    album_page.split_scanned_image()
    assert album_page.found_images == expected_output, Path(input_path).name
