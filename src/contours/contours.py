from typing import Optional

import cv2
import numpy as np


class ContoursFinder:
    # TODO Input: Gets an image and/or image path. Gets (optionally) the number of contours it shall find.
    # TODO Output: Shall return threshold image and a list of the detected contours. And ideally also the detected threshold that was used, if threshold detection was applied.
    # TODO This shall not use the ImageParent class. Also this shouldn't be a utility class but a normal class as then the multiple desired outputs/results can be easier extracted.

    def __init__(
        self,
        image: np.ndarray,
        images_to_detect: Optional[int],
        manual_detection_threshold: int,
        threshold_type: int,  # Could be: cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY
        debug_mode: bool = False,
        write_mode: bool = True,
    ) -> None:
        self.image = image
        self.image_gray = self.transform_into_grayscale()
        self.threshold_image: np.ndarray
        self.images_to_detect = images_to_detect
        self.manual_detection_threshold = manual_detection_threshold
        self.threshold_type = threshold_type
        self.used_threshold: Optional[int]

        self.found_contours: list
        self.found_images = 0  # Number will be incremented once contours of defined size and shape are found.

        self.debug_mode = debug_mode
        self.write_mode = write_mode

    def detect_contours(self):
        self.find_threshold()

    def transform_into_grayscale(self) -> np.ndarray:
        """Transform the image into a grayscale image."""

        image_gray = cv2.cvtColor(src=self.image, code=cv2.COLOR_BGR2GRAY)
        image_gray = cv2.GaussianBlur(src=image_gray, ksize=(3, 3), sigmaX=0)  # Gaussian filtering to remove noise.  # TODO Remove this or make this optional at least.

        return image_gray

    def main(self):  # TODO Find better name.

        if (self.images_to_detect is not None) & (self.manual_detection_threshold == -1):

            found = False

            # Searching for ideal threshold.
            while found is False:
                if self.used_threshold is None:
                    self.used_threshold = 0

                self.used_threshold += 5

                if self.used_threshold > 255:
                    break  # Stops as no (suitable) thresholds could be found.

                self.find_threshold()
                self.find_contours()

        else:
            self.find_threshold()
            self.find_contours()

    def find_threshold(self):
        """Transform the grayscaled image into a binary (black and white) image so that contours can be found best.

        adaptiveThreshold was also tried but decided against as this use case is not about granular extraction of details but to find the corners of scanned images where simple threshold lead to more solid results (as of 2024-03).

        For types of thresholds, see: https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html"""

        if self.manual_detection_threshold >= 0:
            # A threshold is provided by the config.

            self.used_threshold = self.manual_detection_threshold

        elif (self.images_to_detect is not None) & (self.manual_detection_threshold == -1):
            # A threshold shall be found based on the provided number of expected images to detect.

            pass  # No adjustment needed.

        elif (self.images_to_detect is None) & (self.manual_detection_threshold == -1):
            # A threshold will be found based on Otsu binarization.

            self.used_threshold = 0  # Set to 0 as the threshold shall be individually be found by Otsu's binarization.

            self.threshold_type = self.threshold_type + cv2.THRESH_OTSU

        if self.used_threshold is None:
            raise ValueError("self.used_threshold should not be None at this stage.")

        self.threshold = cv2.threshold(
            src=self.image_gray,
            thresh=self.used_threshold,
            maxval=255,
            type=self.threshold_type,
        )[1]

        return self.threshold

    def find_contours(self):
        """Find contours in scanned image that meet size requirements.

        'RETR_EXTERNAL: If you use this flag, it returns only extreme outer flags. All child contours are left behind.'
        More info regarding contour modes, see https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html
        """

        found_contours_raw = cv2.findContours(image=self.threshold, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        found_contours_raw = found_contours_raw[0] if len(found_contours_raw) == 2 else found_contours_raw[1]  # TODO Document why the else part is needed, and in which cases it would trigger.

        # Keep only desired contours.
        self.found_contours = [self._filter_out_too_small_contours(cont) for cont in found_contours_raw]
        self.found_contours = [self._filter_out_contours_with_odd_width_height_ratios(cont) for cont in self.found_contours]
        self.found_contours = [cont for cont in self.found_contours if cont is not None]

        self.found_contours.reverse()  # Reversing the list so the found contours start at the top left and not at the bottom.  # TODO That's not 100% working as intended.

        self.found_images = len(self.found_contours)

        return self.found_contours

    def _filter_out_too_small_contours(self, contour):
        """Remove contours that are smaller than the set pixel threshold."""
        if contour is None:
            return None

        _, _, width, height = cv2.boundingRect(contour)

        # TODO Re-enable
        # if width >= self.min_pixels and height >= self.min_pixels:
        #     return contour
        # else:
        #     return None

    def _filter_out_contours_with_odd_width_height_ratios(self, contour):
        """A scanned picture should have a certain ratio between height and width. Omit contours without those."""
        if contour is None:
            return None

        _, _, width, height = cv2.boundingRect(contour)

        if (
            width / height <= 3 or height / width <= 3
        ):  # Keep only contours with certain width/height ratio. The lower the ratio the more likely false positives. The higher the ratio, the less contours will be filtered by this. # TODO Think about putting this into the config.toml.
            return contour
        else:
            return None

    # TODO Fix
    # def save_found_contours(self, image: np.ndarray, output_path_suffix: str) -> np.ndarray:
    #     """Saves found contours as overlay to the image. This can be used for checks of the set threshold and parameters."""

    #     image = image.copy()

    #     contour_thickness = int(max(self.image_height / 500, self.image_width / 500, 3))  # Dynamicly based on orig image size.

    #     if "threshold" in output_path_suffix:
    #         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Threshold images are grayscale by default and need to be converted to display colored border.

    #     for cont in self.found_contours:
    #         x, y, w, h = cv2.boundingRect(cont)
    #         cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=contour_thickness)

    #     self.save_image(image=image, output_path=self.path_output_stem.parent / (self.path_output_stem.name + "__" + output_path_suffix + self.file_extension))

    #     return image
