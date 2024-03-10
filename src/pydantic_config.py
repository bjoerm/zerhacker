import tomllib
from pathlib import Path

from pydantic import BaseModel


class Paths(BaseModel):
    path_untouched_scans: Path
    path_splitter: Path
    path_fine_cut: Path


class Splitter(BaseModel):
    min_pixel_ratio: float
    detection_threshold: int


class FineCut(BaseModel):
    detection_threshold_finecut: int
    extra_crop: int


class Config(BaseModel):
    paths: Paths
    splitter: Splitter
    fine_cut: FineCut


def config_path() -> Path:
    return Path("src/config.toml")


def load_config() -> Config:
    with config_path().open("rb") as f:
        config = tomllib.load(f)

    config = Config.model_validate(config)

    return config
