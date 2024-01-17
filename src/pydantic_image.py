from pathlib import Path
from typing import Optional

from pydantic import BaseModel, DirectoryPath, FilePath


class Image(BaseModel):
    path: FilePath
    file_name: str
    image: Optional[bytes]
    image_quality: Optional[int]
