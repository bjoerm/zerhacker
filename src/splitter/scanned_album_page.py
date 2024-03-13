from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from shared.image_parent import ImageParent


class ScannedAlbumPage(ImageParent):
    """This class detects and extracts individual images from a single big scanned image."""

    def __init__(self, img_path_input: Path, folder_input: Path, folder_output: Path, manual_threshold: int, min_pixel_ratio: float, debug_mode: bool = False, write_mode: bool = True):
        super().__init__(
            image_path_input=img_path_input, folder_input=folder_input, folder_output=folder_output, debug_mode=debug_mode, write_mode=write_mode
        )  # Executing the __init__ of the parent class.

        self.manual_threshold = manual_threshold

        self.min_pixels = int(min(self.image_height * min_pixel_ratio, self.image_width * min_pixel_ratio))  # Set minimum pixel threshold for filtering out any too small contours.

        self.found_contours: tuple = ()
        self.found_images = 0  # Number will be incremented with each detected images.

    def split_scanned_image(self):
        """This is the main method of this class."""

        self.prepare_image_for_contour_search(manual_threshold=self.manual_threshold)
        self.find_contours()

        if self.debug_mode is True & self.write_mode is True:
            self.save_found_contours(image=self.image, output_path_suffix="contours")
            self.save_found_contours(image=self.threshold, output_path_suffix="contours_threshold")

        for contour in self.found_contours:
            self.extract_and_save_found_image(contour)

        # When no contours are found or only too small contours are detected, copy the image to a special folder in the output folder.
        if self.found_images == 0 & self.write_mode is True:
            self.save_image(
                image=self.image, output_path=self.path_output_stem.parent / Path("no_contour_detected") / (self.path_output_stem.name + "_" + str(self.found_images) + self.file_extension)
            )

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

        _, _, w, h = cv2.boundingRect(contour)

        if w >= self.min_pixels and h >= self.min_pixels:
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

    def extract_and_save_found_image(self, contour) -> np.ndarray:
        """From a single given contour, create a rectangle, extract that rectangle from the scanned image and save it."""

        x, y, w, h = cv2.boundingRect(contour)

        image_cropped = self.image[y : y + h, x : x + w]  # The found cropped image.

        if self.write_mode is True:
            self.save_image(image=image_cropped, output_path=self.path_output_stem.parent / (self.path_output_stem.name + "_" + str(self.found_images) + self.file_extension))

        self.found_images += 1

        return image_cropped
