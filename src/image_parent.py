from pathlib import Path

import cv2
import numpy as np


class ImageParent:
    """Parent class that contains basic operations for dealing with images."""

    def __init__(self, path_input: Path, folder_input: Path, folder_output: Path):
        self.path_input = path_input
        self.path_output_stem, self.path_output_file_extension = self.generate_output_paths(path_input=self.path_input, folder_input=folder_input, folder_output=folder_output)

        self.image_untouched = self.read_image()

    @staticmethod
    def generate_output_paths(path_input: Path, folder_input: Path, folder_output: Path) -> tuple[Path, str]:
        """Generating the parts needed for constructing the output path. This returns the stem and the extension so that possible suffixes to the stem can be added later."""

        path_output = Path(str(path_input).replace(str(folder_input), str(folder_output)))
        path_output_stem = path_output.parent / path_output.stem
        path_output_file_extension = path_output.suffix

        return path_output_stem, path_output_file_extension

    def read_image(self) -> np.ndarray:
        image = cv2.imdecode(np.fromfile(self.path_input, dtype=np.uint8), cv2.IMREAD_UNCHANGED)  # cv2.imread does as of 2021-04 not work for German Umlaute and similar characters. Thus this workaround from: https://stackoverflow.com/a/57872297
        return image


if __name__ == "__main__":

    ImageParent(path_input=Path("pictures/1_untouched_input/01 - Torben/doc10074220210228113627_001.jpg"), folder_input=Path("pictures/1_untouched_input/"), folder_output=Path("pictures/2_rough_cut/"))

    print("End of script reached.")
