from pathlib import Path

import cv2
import numpy as np


class ImageParent:
    """Parent class that contains basic operations needed for images in this case."""

    def __init__(self, img_path_input: Path, folder_input: Path, folder_output: Path):
        self.img_path_input = img_path_input
        self.file_extension = img_path_input.suffix  # TODO Setting this fix to ".png"?
        self.path_output_stem = self.generate_output_paths(path_input=self.img_path_input, folder_input=folder_input, folder_output=folder_output)

        self.image_untouched = self.load_image()

    @staticmethod
    def generate_output_paths(path_input: Path, folder_input: Path, folder_output: Path) -> Path:
        """Generating the parts needed for constructing the output path. This returns the stem and the extension so that possible suffixes to the stem can be added later."""

        path_output = Path(str(path_input).replace(str(folder_input), str(folder_output)))
        path_output_stem = path_output.parent / path_output.stem

        return path_output_stem

    def load_image(self) -> np.ndarray:
        """cv2.imread does as of 2021-04 not work for German Umlaute and similar characters. Thus this workaround from: https://stackoverflow.com/a/57872297"""
        image = cv2.imdecode(np.fromfile(self.img_path_input, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return image

    def save_image(self, image: np.ndarray, output_path: Path):
        """
        Saving image. imwrite does not work with German Umlaute and other special characters. Thus, the following solution.
        Encode the im_resize into the im_buf_cropped, which is a one-dimensional ndarray (from https://jdhao.github.io/2019/09/11/opencv_unicode_image_path/#write-images-with-unicode-paths)
        """

        self.path_output_stem.parent.mkdir(parents=True, exist_ok=True)

        is_success, im_buf_cropped = cv2.imencode(self.file_extension, image)

        if is_success is True:
            im_buf_cropped.tofile(str(output_path))
        else:
            raise ValueError(f"Error when writing file {str(output_path)}.")


if __name__ == "__main__":
    ImageParent(img_path_input=Path("input/01 - T/doc10074220210228113627_001.jpg"), folder_input=Path("input/"), folder_output=Path("output/1_splitter/"))

    print("End of script reached.")
