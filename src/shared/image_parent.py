from pathlib import Path

import cv2
import numpy as np

from shared.logger import logger


class ImageParent:
    """Parent class that contains basic operations needed for images in this case."""

    def __init__(self, image_path_input: Path, folder_input: Path, folder_output: Path, manual_detection_threshold: int, min_pixel_ratio: float, debug_mode: bool = False, write_mode: bool = True):

        self.img_path_input = image_path_input
        self.file_extension = ".png"  # Alternative: image_path_input.suffix. But if that were jpeg, there would be a quality loss due to the multiple read and write steps.
        self.path_output_stem = self.generate_output_paths(path_input=self.img_path_input, folder_input=folder_input, folder_output=folder_output)

        self.image = self.load_image()

        self.image_height: int
        self.image_width: int
        self.update_image_height_and_weight()

        self.manual_detection_threshold = manual_detection_threshold

        self.min_pixels = int(min(self.image_height * min_pixel_ratio, self.image_width * min_pixel_ratio))  # Set minimum pixel threshold for filtering out any too small contours.

        self.found_contours: tuple = ()
        self.found_images = 0  # Number will be incremented with each detected images.

        self.debug_mode = debug_mode
        self.write_mode = write_mode

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

    def update_image_height_and_weight(self) -> tuple[int, int]:
        """Set/update the classes information about image height and width."""
        self.image_height, self.image_width, _ = self.image.shape

        return (self.image_height, self.image_width)

    def transform_into_black_white(self, manual_detection_threshold: int = -1) -> np.ndarray:
        """Transform the image into a grayscale image and then into a binary (black and white) image so that contours can be found best.

        For types of thresholds, see: https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html"""

        image_gray = cv2.cvtColor(src=self.image, code=cv2.COLOR_BGR2GRAY)
        image_gray = cv2.GaussianBlur(src=image_gray, ksize=(5, 5), sigmaX=0)  # Gaussian filtering to remove noise.  # TODO Maybe have this as input as it might not be needed for the rotation part.

        if manual_detection_threshold >= 0:
            threshold_type = cv2.THRESH_BINARY_INV
            threshold_value = manual_detection_threshold
        else:  # Automatic threshold estimation.
            threshold_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU  # THRESH_OTSU will automatically find pretty good thresholds.
            threshold_value = 0  # Set to 0 as the threshold shall be individually be found by Otsu's Binarization in the following step.

        self.threshold = cv2.threshold(
            src=image_gray,
            thresh=threshold_value,
            maxval=255,
            type=threshold_type,
        )[1]

        return self.threshold

    def find_contours(self):
        """Find (rectangle) contours in scanned image that meet size requirements.

        'RETR_EXTERNAL: If you use this flag, it returns only extreme outer flags. All child contours are left behind.'
        More info regarding contour modes, see https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html
        """

        self.found_contours = cv2.findContours(image=self.threshold, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        self.found_contours = self.found_contours[0] if len(self.found_contours) == 2 else self.found_contours[1]  # TODO Document why the else part is needed, and in which cases it would trigger.

        if len(self.found_contours) >= 1:
            pass
        else:
            raise ValueError(f"No contour was found in the image: str({self.img_path_input})")

        # Keep only desired contours.
        self.found_contours = [self._filter_out_too_small_contours(cont) for cont in self.found_contours]
        self.found_contours = [self._filter_out_contours_with_odd_width_height_ratios(cont) for cont in self.found_contours]
        self.found_contours = [cont for cont in self.found_contours if cont is not None]

        if len(self.found_contours) >= 1:
            pass
        else:
            raise ValueError(f"No contour was found in the image: str({self.img_path_input})")

        self.found_contours.reverse()  # Reversing the list so the found contours start at the top left and not at the bottom.  # TODO That's not 100% working as intended.

        return self.found_contours

    def _filter_out_too_small_contours(self, contour):
        """Remove contours that are smaller than the set pixel threshold."""
        if contour is None:
            return None

        _, _, width, height = cv2.boundingRect(contour)

        if width >= self.min_pixels and height >= self.min_pixels:
            return contour
        else:
            return None

    def _filter_out_contours_with_odd_width_height_ratios(self, contour):
        """A scanned picture should have a certain ratio between height and width. Omit contours without those."""
        if contour is None:
            return None

        _, _, width, height = cv2.boundingRect(contour)

        if (
            width / height <= 3 or height / width <= 3
        ):  # The lower the ratio the more likely false positives. The higher the ratio, the less contours will be filtered by this. # TODO Think about putting this into the config.toml.
            return contour
        else:
            return None

    def save_found_contours(self, image: np.ndarray, output_path_suffix: str) -> np.ndarray:
        """Saves found contours as overlay to the image. This can be used for checks of the set threshold and parameters."""

        image = image.copy()

        contour_thickness = int(max(self.image_height / 500, self.image_width / 500, 3))  # Dynamicly based on orig image size.

        # Threshold images are grayscale by default and need to be converted to display colored border.
        if "threshold" in output_path_suffix:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        for cont in self.found_contours:
            x, y, w, h = cv2.boundingRect(cont)
            cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=contour_thickness)

        self.save_image(image=image, output_path=self.path_output_stem.parent / (self.path_output_stem.name + "__" + output_path_suffix + self.file_extension))

        return image


if __name__ == "__main__":
    ImageParent(image_path_input=Path("input/01 - T/doc10074220210228113627_001.jpg"), folder_input=Path("input/"), folder_output=Path("output/1_splitter/"))

    print("End of script reached.")
