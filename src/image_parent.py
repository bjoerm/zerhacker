from pathlib import Path

import cv2
import numpy as np


class ImageParent:
    """Parent class that contains basic operations needed for images in this case."""

    def __init__(self, image_path_input: Path, folder_input: Path, folder_output: Path, debug_mode: bool = False):
        self.img_path_input = image_path_input
        self.file_extension = ".png"  # Alternative: image_path_input.suffix. But if that were jpeg, there would be a quality loss due to the multiple read and write steps.
        self.path_output_stem = self.generate_output_paths(path_input=self.img_path_input, folder_input=folder_input, folder_output=folder_output)

        self.image = self.load_image()

        self.image_height, self.image_width, _ = self.image.shape

        self.debug_mode = debug_mode

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

    def prepare_image_for_contour_search(self):
        """Transform the image into a black and white image so that contours can be found best. For types of thresholds, see: https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html"""

        image_gray = cv2.cvtColor(src=self.image, code=cv2.COLOR_BGR2GRAY)
        image_blurred = cv2.GaussianBlur(src=image_gray, ksize=(5, 5), sigmaX=0)  # Gaussian filtering to remove noise.

        self.threshold = cv2.threshold(
            src=image_blurred,
            thresh=0,  # Set to 0 as the threshold shall be individually be found by Otsu's Binarization.
            maxval=255,
            type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,  #
        )[1]


if __name__ == "__main__":
    ImageParent(image_path_input=Path("input/01 - T/doc10074220210228113627_001.jpg"), folder_input=Path("input/"), folder_output=Path("output/1_splitter/"))

    print("End of script reached.")
