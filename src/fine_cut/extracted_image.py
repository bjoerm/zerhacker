import cv2
import numpy as np

from shared.image_parent import ImageParent
from shared.logger import logger


class ExtractedImage(ImageParent):
    """Rotate a single image and crop the border to remove possible remains from the scanning.

    This is a separate module from the inital splitting in order to allow for feeding of manual cut images for this fine cut.
    """

    def rotate_and_crop(self, extra_crop: int):
        """This is the main method of this class."""

        logger.debug(f"Start rotate_and_crop class: {self.img_path_input}")

        self.add_white_border()

        self.transform_into_black_white(manual_detection_threshold=self.manual_detection_threshold)
        self.find_contours()

        if len(self.found_contours) != 1:
            logger.info(f"No single contour found in: {self.img_path_input}")
            return

        if self.debug_mode is True & self.write_mode is True:
            self.save_found_contours(image=self.image, output_path_suffix="DEBUG2_contours")
            self.save_found_contours(image=self.threshold, output_path_suffix="DEBUG2_contours_threshold")

        self.fine_rotate_image()
        self.crop_border(extra_crop=extra_crop)

        if self.write_mode is True:
            self.save_image(image=self.image, output_path=self.path_output_stem.parent / (self.path_output_stem.name + self.file_extension))

    def add_white_border(self):
        """Add white pixels around the image. This will help for the image rotation part."""

        additional_border = 300

        self.image = cv2.copyMakeBorder(
            src=self.image,
            top=additional_border,
            bottom=additional_border,
            left=additional_border,
            right=additional_border,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        self.update_image_height_and_weight()  # Update the values.

    def fine_rotate_image(self):
        """Detects the best (rotated) rectangle from the largest contour. This is then used for a fine rotation. This is required if scanned images were slighty rotated."""

        image = self.image.copy()

        largest_contour = max(self.found_contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        logger.debug(f"Detected degree for rotation: {rect[2]}")  # TODO This currently rotates oddly. Fix this!

        width = int(rect[1][0])
        height = int(rect[1][1])

        box = cv2.boxPoints(rect)
        box = np.int64(box)

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (width, height))

        # Rather dirty hack to prevent the rotation from minAreaRect to go overboard.
        if rect[2] > 45:  # This is the degree (should range from 0 to 90), when the image is rotated "back".
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            logger.debug(f"Rotate image back: {self.img_path_input}")

        else:
            pass

        self.image = warped
        self.update_image_height_and_weight()

    def crop_border(self, extra_crop: int):

        cropped_img = self.image[extra_crop : self.image_height - extra_crop, extra_crop : self.image_width - extra_crop]  # top_border:bottom_border, left_border:right_border

        self.image = cropped_img
        self.update_image_height_and_weight()

        # cropped_img = cv2.resize(cropped_img, (target_width, target_height))
