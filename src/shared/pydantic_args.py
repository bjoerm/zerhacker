from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class Args(BaseModel):
    """Pydantic class for the arguments."""

    task: Literal["split", "fine_cut"]
    image_path_input: Path
    folder_input: Path
    folder_output: Path
    manual_detection_threshold: int
    min_pixel_ratio: float
    debug_mode: bool
    write_mode: bool
    extra_crop: int
