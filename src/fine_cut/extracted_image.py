from pathlib import Path

import cv2

from shared.image_parent import ImageParent


class ExtractedImage(ImageParent):
    """Automatically rotate a single image and crop the border to remove possible remains from the scanning."""

    def __init__(self, img_path_input: Path, folder_input: Path, folder_output: Path, extra_crop: int, debug_mode: bool = False, write_mode: bool = True):
        super().__init__(
            image_path_input=img_path_input, folder_input=folder_input, folder_output=folder_output, debug_mode=debug_mode, write_mode=write_mode
        )  # Executing the __init__ of the parent class.

        self.extra_crop = extra_crop

    def rotate_and_crop(self):
        self.add_white_border()
        self.find_quadrilateral()
        self.expand_quadrilateral_to_rectangle()
        self.crop_border()
        self.rotate_image()

    def add_white_border(self):
        """Add white pixels around the image. This will help for the image rotation part."""

        self.image = cv2.copyMakeBorder(
            src=self.image,
            top=100,  # TODO The number of pixels might be limiting for huge images that need 45 degree rotation.
            bottom=100,
            left=100,
            right=100,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        self.get_image_height_and_weight()  # Update the values.

    def find_quadrilateral(self):
        # TODO Was:
        # TODO cv2.threshold(was on gray, but the other approach from the splitter should be more accurate).
        # TODO cv2.findContours.
        # TODO cv2.contourArea
        # TODO If the found contourArea's size (in pixel sum) is at least between 1% to 99% of original area of the image, then do:
        ## TODO Contour Perimeter (cv2.arcLength) / 10 as epsilon for cv2.approxPolyDP()

        # TODO Alternative: self.prepare_image_for_contour_search()

        self.prepare_image_for_contour_search(manual_threshold=-1)

        # TODO find_contours from ScannedAlbumPage into ImageGenerator and use it here? Same for _filter_out_too_small_contours,_filter_out_contours_with_odd_width_height_ratios and save_found_contours which should also be valueable here?

    def expand_quadrilateral_to_rectangle(self):
        pass

    def rotate_image(self):
        pass

    def crop_border(self):
        pass
