from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from shared.image_parent import ImageParent


class ContoursFinder:
    # TODO Input: Gets an image and/or image path. Gets (optionally) the number of contours it shall find.
    # TODO Output: Shall return threshold image and a list of the detected contours. And ideally also the detected threshold that was used, if threshold detection was applied.
    # TODO This shall not use the ImageParent class. Also this shouldn't be a utility class but a normal class as then the multiple desired outputs/results can be easier extracted.

    def __init__(self, image: np.ndarray, images_to_detect: Optional[int], manual_detection_threshold: int, debug_mode: bool = False, write_mode: bool = True) -> None:
        self.image = image
        self.threshold_image: np.ndarray
        self.images_to_detect = images_to_detect
        self.manual_detection_threshold = manual_detection_threshold
        self.used_threshold: int

        self.found_contours: tuple = ()
        self.found_images = 0  # Number will be incremented with each detected images.

        self.debug_mode = debug_mode
        self.write_mode = write_mode

    def detect_contours(cls, image: np.ndarray) -> dict:
        pass

    @classmethod
    def find_threshold(cls, manual_detection_threshold: int = -1):
        pass

    def prepare_image_for_contour_search(self, manual_detection_threshold: int = -1) -> np.ndarray:
        """Transform the image into a grayscale image and then into a binary (black and white) image so that contours can be found best.

        For types of thresholds, see: https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html"""

        image_gray = cv2.cvtColor(src=self.image, code=cv2.COLOR_BGR2GRAY)
        image_blurred = cv2.GaussianBlur(src=image_gray, ksize=(5, 5), sigmaX=0)  # Gaussian filtering to remove noise.

        if manual_detection_threshold >= 0:
            threshold_type = cv2.THRESH_BINARY_INV
            threshold_value = manual_detection_threshold
        else:  # Automatic threshold estimation.
            threshold_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU  # THRESH_OTSU will automatically find pretty good thresholds.
            threshold_value = 0  # Set to 0 as the threshold shall be individually be found by Otsu's Binarization.

        self.threshold = cv2.threshold(
            src=image_blurred,
            thresh=threshold_value,
            maxval=255,
            type=threshold_type,
        )[1]

        return self.threshold

    def find_contours(self):
        """Find contours in scanned image that meet size requirements.

        'RETR_EXTERNAL: If you use this flag, it returns only extreme outer flags. All child contours are left behind.'
        More info regarding contour modes, see https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html
        """

        self.found_contours = cv2.findContours(image=self.threshold, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        self.found_contours = self.found_contours[0] if len(self.found_contours) == 2 else self.found_contours[1]  # TODO Document why the else part is needed, and in which cases it would trigger.

        # Keep only desired contours.
        self.found_contours = [self._filter_out_too_small_contours(cont) for cont in self.found_contours]
        self.found_contours = [self._filter_out_contours_with_odd_width_height_ratios(cont) for cont in self.found_contours]
        self.found_contours = [cont for cont in self.found_contours if cont is not None]

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

        if "threshold" in output_path_suffix:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Threshold images are grayscale by default and need to be converted to display colored border.

        for cont in self.found_contours:
            x, y, w, h = cv2.boundingRect(cont)
            cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=contour_thickness)

        self.save_image(image=image, output_path=self.path_output_stem.parent / (self.path_output_stem.name + "__" + output_path_suffix + self.file_extension))

        return image
