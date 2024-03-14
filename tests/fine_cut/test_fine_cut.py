import shutil
from pathlib import Path

import pytest

from fine_cut.extracted_image import ExtractedImage
from shared.pydantic_config import load_config


@pytest.fixture
def write_mode():
    """Central switch for writing files into the path_folder_temp_output folder."""
    return True


@pytest.fixture
def path_folder_input():
    return Path("tests/fine_cut/input")


@pytest.fixture
def path_folder_temp_output():
    return Path("tests/fine_cut/temp_output")


@pytest.fixture
def config():
    return load_config()


def test_remove_temp_output_folder(path_folder_temp_output):
    """Tidying up before launching new tests."""
    shutil.rmtree(path_folder_temp_output, ignore_errors=True)


def test_integration_fine_cut(config, path_folder_input, path_folder_temp_output, write_mode):

    images = list(path_folder_input.iterdir())

    for file in images:
        extracted_image = ExtractedImage(
            image_path_input=file,
            folder_input=path_folder_input,
            folder_output=path_folder_temp_output,
            manual_threshold=config.fine_cut.manual_detection_threshold,
            min_pixel_ratio=config.fine_cut.min_pixel_ratio,
            debug_mode=True,
            write_mode=write_mode,
        )

        extracted_image.rotate_and_crop(extra_crop=config.fine_cut.extra_crop)
