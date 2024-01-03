import os
from pathlib import Path
from typing import List

import cv2
import numpy as np


class SharedUtility:
    """This class contains shared utility methods."""

    @classmethod
    def generate_file_list(cls, path: str) -> List[str]:
        """Generate a list of all images in the input path."""

        file_list_gen = Path(path)
        file_list = [str(f.parent / f.name) for f in file_list_gen.rglob("*.jpg")]  # rglob is recursive.

        # # TODO Example of how to deal with multiple file extensions:
        # types = ("*.jpg", "*.JPG", "*.JPEG", "*.jpeg", "*.png", "*.PNG")  # TODO use this approach in the other class as well. But then also fix the the file output there, which is hard coded to .jpg.

        # for t in types:
        #     if glob.glob(f"{input_path}/{t}") != []:  # TODO Should I add here ** for recursive search?
        #         f_l = glob.glob(f"{input_path}/{t}")
        #         for f in f_l:
        #             files.append(f)

        file_list = cls._filter_out_debug_files(file_list)

        return file_list

    @staticmethod
    def _filter_out_debug_files(files: List[str]):
        """Removing files that are only used for debugging/reviewing."""

        files = [i for i in files if not ("__debug_" in i)]  # Identifier is the substring "__debug_"

        return files

    @staticmethod
    def get_available_threads() -> int:
        """Get number of available CPU threads that can be used for parallel processing."""

        try:
            num_threads = os.cpu_count()  # This is not 100% the best approach but os.sched_getaffinity does not work on Windows.

        except NotImplementedError:
            print("Automatic thread detection didn't work. Defaulting to 1 thread only.")
            num_threads = 1

        print(f"\n[Status] Using {num_threads} threads.")

        return num_threads

    @staticmethod
    def save_image(image: np.ndarray, output_path: Path, jpg_quality: int):
        """
        Saving image. imwrite does not work with German Umlaute and other special characters. Thus, the following solution.
        Encode the im_resize into the im_buf_cropped, which is a one-dimensional ndarray (from https://jdhao.github.io/2019/09/11/opencv_unicode_image_path/#write-images-with-unicode-paths)
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)

        is_success, im_buf_cropped = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])

        if is_success is True:
            im_buf_cropped.tofile(str(output_path))
        else:
            raise ValueError("Errer when writing file.")
