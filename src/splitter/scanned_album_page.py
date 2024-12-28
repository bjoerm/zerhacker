from pathlib import Path

import cv2
import numpy as np

from shared.image_parent import ImageParent
from shared.logger import logger


class ScannedAlbumPage(ImageParent):
    """This class detects and extracts individual images from a single big scanned image."""

    def split_scanned_image(self):
        """This is the main method of this class."""

        logger.debug(f"Start split_scanned_image class: {self.img_path_input}")

        self.transform_into_black_white(manual_detection_threshold=self.manual_detection_threshold)
        self.find_contours()

        if self.debug_mode is True & self.write_mode is True:
            self.save_found_contours(image=self.image, output_path_suffix="DEBUG1_contours")
            self.save_found_contours(image=self.threshold, output_path_suffix="DEBUG1_contours_threshold")

        for contour in self.found_contours:
            self.extract_and_save_found_image(contour)

    def extract_and_save_found_image(self, contour) -> np.ndarray:
        """From a single given contour, create a rectangle, extract that rectangle from the scanned image and save it."""

        x, y, w, h = cv2.boundingRect(contour)

        image_cropped = self.image[y : y + h, x : x + w]  # The found cropped image.

        if self.write_mode is True:
            self.save_image(image=image_cropped, output_path=self.path_output_stem.parent / (self.path_output_stem.name + "_" + str(self.found_images) + self.file_extension))

        self.found_images += 1

        return image_cropped
