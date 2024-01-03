import shutil
import tomllib

from pydantic_config import Config

with open("src/config.toml", "rb") as f:
    config = tomllib.load(f)

config = Config(**config)

for i in [config.paths.path_rough_cut, config.paths.path_fine_cut]:
    shutil.rmtree(i, ignore_errors=True)

for key, pth in config.paths:
    pth.mkdir(parents=True, exist_ok=True)

file_list = [str(f.parent / f.name) for f in config.paths.path_untouched_scans.rglob("*.jpg")]  # rglob is recursive.
