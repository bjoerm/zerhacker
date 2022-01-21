from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import tqdm

from shared_utils import SharedUtility


class Splitter:
    """This class detects and extracts individual images from a big scanned image."""

    def __init__(self, params: dict):
        self.input_image_path = params.get("input_image_path")
        self.output_image_path = str(self.input_image_path).replace(
            params.get("input_path"), params.get("output_path")
        )

        self.original = cv2.imdecode(
            np.fromfile(self.input_image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )  # cv2.imread does as of 2021-04 not work for German Umlaute and similar characters. From: https://stackoverflow.com/a/57872297

        self.min_pixels = params.get("min_pixels")
        self.detection_threshold = params.get("detection_threshold")
        self.jpg_quality = 95  # TODO Ideally use the same quality that the input file had, if this is saved in a jpg file when saving.

    def prepare_image(self):
        """Transform the image so that contours can best be found."""

        gray = cv2.cvtColor(src=self.original, code=cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(src=gray, ksize=(3, 3), sigmaX=0)

        self._thresh = cv2.threshold(
            src=blurred,
            thresh=self.detection_threshold,
            maxval=255,
            type=cv2.THRESH_BINARY_INV,  # For values on thresholds, see: https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
        )[1]

    def find_contours(self):
        """Find contours"""
        self.prepare_image()

        self._contours = cv2.findContours(
            image=self._thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )

        self._contours = (
            self._contours[0] if len(self._contours) == 2 else self._contours[1]
        )

        self._contours.reverse()  # Reversing the list so the found contours start at the top left and not at the bottom.

        # TODO test = [self._filter_out_contours(c) for c in self._contours]  # TODO this is not yet used and it also would need to be expanded to deal with the returned Nones properly.

    def _filter_out_contours(self, contour):
        """Remove contours that are smaller than the set pixel threshold."""

        x, y, w, h = cv2.boundingRect(contour)

        if w < self.min_pixels or h < self.min_pixels:
            return None
        else:
            return contour

    def extract_single_images(self):
        """TODO"""

        self.find_contours()

        # Iterate through contours and filter for cropped
        self._found_images = 0

        found_pictures = self.original.copy()

        for c in self._contours:
            # TODO Split this into multiple smaller functions.

            x, y, w, h = cv2.boundingRect(c)

            if w < self.min_pixels or h < self.min_pixels:
                continue  # If the detected rectangle is smaller than x min_pixels in either width or height, skip this rectangle/contour.

            else:
                # Mark on the big scanned image all found single images.
                found_pictures = cv2.rectangle(
                    img=found_pictures,
                    pt1=(x, y),
                    pt2=(x + w, y + h),
                    color=(36, 255, 12),
                    thickness=25,
                )

                cropped = self.original[
                    y : y + h, x : x + w
                ]  # The found cropped image.

                # Ensure that (sub) folders for the respective album exist. (Is required if there are folders in the input.)
                Path(self.output_image_path).parent.mkdir(parents=True, exist_ok=True)

                # Saving image. imwrite does not work with German Umlauten and other special characters. Thus, the following solution.
                # Encode the im_resize into the im_buf_cropped, which is a one-dimensional ndarray (from https://jdhao.github.io/2019/09/11/opencv_unicode_image_path/#write-images-with-unicode-paths)
                is_success, im_buf_cropped = cv2.imencode(
                    ".jpg", cropped, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality]
                )

                if is_success is True:
                    im_buf_cropped.tofile(
                        self.output_image_path.replace(
                            ".jpg", f"_cr_{self._found_images}.jpg"
                        )
                    )

                else:
                    print("WARNING File could not be read.")

                self._found_images += 1

        # When no contours are found or only too small contours are detected, copy the image to a special folder in the output folder.
        if self._found_images == 0:
            (Path(self.output_image_path).parent / Path("no_crop_done")).mkdir(
                parents=True, exist_ok=True
            )

            # TODO The saving part is very inconsistent. The part below uses a different approach then the saving above .tofile().

            # Saving image. imwrite does not work with German Umlaute and other special characters. Thus, the following solution.
            # Encode the im_resize into the im_buf_cropped, which is a one-dimensional ndarray (from https://jdhao.github.io/2019/09/11/opencv_unicode_image_path/#write-images-with-unicode-paths)
            is_success, im_buf_cropped = cv2.imencode(
                ".jpg", self.original, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality]
            )

            if is_success is True:
                im_buf_cropped.tofile(
                    str(
                        Path(self.output_image_path).parent
                        / Path("no_crop_done")
                        / Path(self.input_image_path).name
                    )
                )
            else:
                print("WARNING File could not be read.")
        else:
            # Mark on the big scanned image all found single images.
            cv2.imwrite(
                filename=str(self.output_image_path).replace(".jpg", "__DEBUG.jpg"),
                img=found_pictures,
            )

        return


def start_splitting(
    parent_path_images: str,
    input_path: str,
    output_path: str,
    min_pixels: int,
    detection_threshold: int,
    num_threads: int,
):
    """Split a list of images. Uses multiple CPU threads to split faster."""

    print("\n[Status] Started Splitter.")

    files = SharedUtility.generate_file_list(
        path=Path(parent_path_images) / Path(input_path)
    )

    if len(files) == 0:
        print(
            f"No image files found in {input_path}\n Exiting."
        )  # TODO Add stop of the programm.

    else:
        # Creating list of dictionaries for parallel processing.
        params = []

        for f in files:
            params.append(
                {
                    "input_image_path": f,
                    "input_path": input_path,
                    "output_path": output_path,
                    "min_pixels": min_pixels,
                    "detection_threshold": detection_threshold,
                }
            )

        with Pool(num_threads) as p:
            list(tqdm.tqdm(p.imap(_call_splitter, params), total=len(params)))

    print("\n[Status] Finished Splitter.")


def _call_splitter(params: dict):
    """Function to be called by multiple threads in parallel. See also https://stackoverflow.com/a/21345308 for another parallel package."""
    test = Splitter(params)
    test.extract_single_images()


if __name__ == "__main__":
    import toml

    from environment import Environment

    # Load options
    cfg = toml.load("options.toml", _dict=dict)

    Environment.initiate(
        cfg.get("parent_path_images"),
        cfg.get("untouched_scans_path"),
        cfg.get("rough_cut_path"),
        cfg.get("fine_cut_path"),
    )

    cfg["num_threads"] = SharedUtility.get_available_threads()

    start_splitting(
        parent_path_images=cfg.get("parent_path_images"),
        input_path=cfg.get("untouched_scans_path"),
        output_path=cfg.get("rough_cut_path"),
        min_pixels=cfg.get("min_pixels"),
        detection_threshold=cfg.get("detection_threshold"),
        num_threads=cfg.get("num_threads"),
    )
