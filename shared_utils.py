import os
from pathlib import Path
import numpy as np
import cv2


class SharedUtility:
    """This class contains shared utility methods."""

    @staticmethod
    def generate_file_list(path: str) -> list:
        """Generate a list of all images in the input path."""

        file_list_gen = Path(path)
        file_list = [str(f.parent / f.name) for f in file_list_gen.rglob("*.jpg")]  # rglob is recursive. # TODO Deal with .jpeg, .JPG, ...

        # # TODO Example of how to deal with multiple file extensions:
        # types = ("*.jpg", "*.JPG", "*.JPEG", "*.jpeg", "*.png", "*.PNG")  # TODO use this approach in the other class as well. But then also fix the the file output there, which is hard coded to .jpg.

        # for t in types:
        #     if glob.glob(f"{input_path}/{t}") != []:  # TODO Should I add here ** for recursive search?
        #         f_l = glob.glob(f"{input_path}/{t}")
        #         for f in f_l:
        #             files.append(f)

        return file_list

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
