from pathlib import Path

import cv2
import numpy as np

from shared.image_parent import ImageParent


class ScannedAlbumPage(ImageParent):
    """This class detects and extracts individual images from a single big scanned image."""

    def split_scanned_image(self):
        """This is the main method of this class."""

        self.prepare_image_for_contour_search(manual_detection_threshold=self.manual_detection_threshold)
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

    def extract_and_save_found_image(self, contour) -> np.ndarray:
        """From a single given contour, create a rectangle, extract that rectangle from the scanned image and save it."""

        x, y, w, h = cv2.boundingRect(contour)

        image_cropped = self.image[y : y + h, x : x + w]  # The found cropped image.

        if self.write_mode is True:
            self.save_image(image=image_cropped, output_path=self.path_output_stem.parent / (self.path_output_stem.name + "_" + str(self.found_images) + self.file_extension))

        self.found_images += 1

        return image_cropped
