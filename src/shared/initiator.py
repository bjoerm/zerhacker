import shutil
from pathlib import Path

from pydantic import DirectoryPath
from pydantic.dataclasses import dataclass

from shared.logger import logger


@dataclass
class Initiator:
    input_folder: DirectoryPath
    output_folder: Path

    def init(self) -> list[Path]:
        """Prepare the output folder and return all relevant files from the input folder."""

        logger.debug("Start initiator class.")

        shutil.rmtree(self.output_folder, ignore_errors=True)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        file_list = [f for f in self.input_folder.rglob("*.jpg")] + [f for f in self.input_folder.rglob("*.png")]  # The "r" in rglob is for recursive.

        file_list = [f for f in file_list if f.name.find("DEBUG") == -1]  # Ignoring the files that are created in DEBUG mode.

        if len(file_list) == 0:
            raise ValueError("No relevant files detected in input folder.")

        logger.debug("End initiator class.")

        return file_list
