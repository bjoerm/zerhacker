import shutil
from pathlib import Path
from typing import Optional

from pydantic import DirectoryPath
from pydantic.dataclasses import dataclass


@dataclass
class Initiator:
    input_folder: DirectoryPath
    output_folder: Path

    def init(self) -> list[Path]:
        shutil.rmtree(self.output_folder, ignore_errors=True)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        file_list = [f for f in self.input_folder.rglob("*.jpg")]  # rglob is recursive. # TODO Expand from only using .jpg.

        return file_list
