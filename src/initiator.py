import shutil
from pathlib import Path

from pydantic import DirectoryPath
from pydantic.dataclasses import dataclass


@dataclass
class Initiator:
    input_folder: DirectoryPath
    output_folder: Path

    def init(self) -> list[Path]:
        """Prepare the output folder and return all relevant files from the input folder."""

        shutil.rmtree(self.output_folder, ignore_errors=True)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        file_list = [f for f in self.input_folder.rglob("*.jpg")]  # rglob is recursive. # TODO Expand from only using .jpg.

        if len(file_list) == 0:
            raise ValueError("No relevant files detected in input folder.")

        return file_list
