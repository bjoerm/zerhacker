from pathlib import Path
from typing import Optional

from pydantic import BaseModel, DirectoryPath, FilePath


class ImageParent(BaseModel):
    path: FilePath
    folder_input: DirectoryPath
    folder_output: DirectoryPath


class ScannedPage(BaseModel):
    path: FilePath
    file_name: str
    image_untouched: Optional[bytes] = None  # TODO Do I really want to save the original? Or is this paranoia?
    image_touched: Optional[bytes] = None
    jpg_quality: Optional[int] = None


class Image(BaseModel):
    path: FilePath
    file_name: str
    image_untouched: Optional[bytes]  # TODO Do I really want to save the original? Or is this paranoia?
    image_touched: Optional[bytes]
    jpg_quality: Optional[int]


print("test")
