from pathlib import Path

from pydantic import BaseModel


class Album(BaseModel):
    """A set of image files."""

    path: Path
    files: list[Path]


class Image(BaseModel):
    path: Path
    file_name: str
    image_untouched: bytes  # TODO Do I really want to save the original? Or is this paranoia?
    image_touched: bytes
