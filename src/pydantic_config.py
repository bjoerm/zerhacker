from pathlib import Path

from pydantic import BaseModel, DirectoryPath, FilePath


class Paths(BaseModel):
    path_untouched_scans: Path
    path_rough_cut: Path
    path_fine_cut: Path


class Splitter(BaseModel):
    min_pixel_ratio: float
    detection_threshold: int


class Finecut(BaseModel):
    detection_threshold_finecut: int
    extra_crop: int


class Config(BaseModel):
    paths: Paths
    splitter: Splitter
    finecut: Finecut
