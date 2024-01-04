from pathlib import Path

from pydantic import BaseModel, FilePath


class Image(BaseModel):
    path: FilePath
    file_name: str
    image_untouched: bytes  # TODO Do I really want to save the original? Or is this paranoia?
    image_touched: bytes
    jpg_quality: int
