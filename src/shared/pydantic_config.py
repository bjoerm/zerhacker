import tomllib
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class General(BaseModel):
    debug_mode: bool
    write_mode: bool


class Paths(BaseModel):
    path_untouched_scans: Path
    path_splitter: Path
    path_fine_cut: Path


class Splitter(BaseModel):
    manual_detection_threshold: int
    min_pixel_ratio: float


class FineCut(BaseModel):
    manual_detection_threshold: int
    min_pixel_ratio: float
    extra_crop: int


class Parallel(BaseModel):
    num_threads: Optional[int]


class Config(BaseModel):
    general: General
    paths: Paths
    splitter: Splitter
    fine_cut: FineCut
    parallel: Parallel


def config_path() -> Path:
    return Path("src/config.toml")


def load_config() -> Config:
    with config_path().open("rb") as f:
        config = tomllib.load(f)

    config = Config.model_validate(config)

    return config
