import cv2

from shared.image_parent import ImageParent


class ExtractedImage(ImageParent):
    """Automatically rotate a single image and crop the border to remove possible remains from the scanning."""

    def rotate_and_crop(self, extra_crop: int):
        self.add_white_border()
        self.find_quadrilateral()
        self.expand_quadrilateral_to_rectangle()
        self.crop_border(extra_crop=extra_crop)
        self.rotate_image()

        if self.debug_mode is True & self.write_mode is True:
            self.save_found_contours(image=self.image, output_path_suffix="contours")
            self.save_found_contours(image=self.threshold, output_path_suffix="contours_threshold")

        if self.write_mode is True:
            self.save_image(image=self.image, output_path=self.path_output_stem.parent / (self.path_output_stem.name + "_" + str(self.found_images) + self.file_extension))

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

        self.prepare_image_for_contour_search(manual_detection_threshold=self.manual_detection_threshold)
        self.find_contours()

    def expand_quadrilateral_to_rectangle(self):
        pass

    def rotate_image(self):
        pass

    def crop_border(self, extra_crop: int):
        pass
