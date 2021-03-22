import cv2
import glob
from pathlib import Path


class MultiCrop:
    """ TODO """

    @classmethod
    def split_scanned_image(cls, input_path, output_path, error_path, min_pixels):
        """ TODO """

        file_list = cls._generate_file_list(input_path)

        # cls._split_scanned_image(image, output_path, error_path, min_pixels)

    @staticmethod
    def _generate_file_list(input_path: str) -> list:
        """ Generate a list of all images in the input path. """ # TODO Put this into a util class as this will be used by both the MultiCrop as well as the FineCut class.

        file_list = glob.glob(input_path + "/**", recursive=True)

        return(file_list)