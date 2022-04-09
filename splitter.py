from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import tqdm

from shared_utils import SharedUtility


class Splitter:
    """This class detects and extracts individual images from a single big scanned image."""

    def __init__(self, params: dict):
        self.input_image_path = Path(params.get("input_image_path"))
        self.output_image_path = Path(str(self.input_image_path).replace(params.get("input_path"), params.get("output_path")))
        self.output_no_contour_path = Path(self.output_image_path).parent / Path("no_contour_detected") / Path(self.input_image_path).name
        self.output_contour_debug_path = Path(self.output_image_path).parent / Path("contour_debug") / Path(self.input_image_path).name

        self.img_original = cv2.imdecode(np.fromfile(self.input_image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)  # cv2.imread does as of 2021-04 not work for German Umlaute and similar characters. From: https://stackoverflow.com/a/57872297

        self.min_pixels = params.get("min_pixels")
        self.detection_threshold = params.get("detection_threshold")
        self.jpg_quality = 95  # TODO Ideally use the same quality that the input file had, if this is saved in a jpg file when saving.

    def split_scanned_image(self):
        """This is the main method of this class."""

        self.find_contours()

        self.save_found_contours()

        self.found_images = 0

        for contour in self._contours:
            self.extract_and_save_found_image(contour)

        # When no contours are found or only too small contours are detected, copy the image to a special folder in the output folder.
        if self.found_images == 0:
            SharedUtility.save_image(self.img_original, self.output_no_contour_path, self.jpg_quality)

    def find_contours(self):
        """Find contours in scanned image that meet size requirements."""
        self._prepare_image_for_contour_search()

        self._contours = cv2.findContours(image=self._thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        self._contours = self._contours[0] if len(self._contours) == 2 else self._contours[1]

        self._contours.reverse()  # Reversing the list so the found contours start at the top left and not at the bottom.

        # Filter out too small contours
        self._contours = [self._filter_out_too_small_contours(c) for c in self._contours]
        self._contours = [i for i in self._contours if i is not None]

    def _prepare_image_for_contour_search(self):
        """Transform the image so that contours can be found best."""

        img_gray = cv2.cvtColor(src=self.img_original, code=cv2.COLOR_BGR2GRAY)
        img_blurred = cv2.GaussianBlur(src=img_gray, ksize=(3, 3), sigmaX=0)

        self._thresh = cv2.threshold(
            src=img_blurred,
            thresh=self.detection_threshold,
            maxval=255,
            type=cv2.THRESH_BINARY_INV,  # For values on thresholds, see: https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
        )[1]

    def _filter_out_too_small_contours(self, contour):
        """Remove contours that are smaller than the set pixel threshold."""

        x, y, w, h = cv2.boundingRect(contour)

        if w >= self.min_pixels and h >= self.min_pixels:
            return contour
        else:
            return None

    def save_found_contours(self):
        """Saves found contours as overlay to the image (in a separate output folder). This can be used for checks of the set thresholds and parameters."""

        pic_with_contours = self.img_original.copy()
        cv2.drawContours(image=pic_with_contours, contours=self._contours, contourIdx=-1, color=(0, 255, 0), thickness=15, lineType=cv2.LINE_AA)

        SharedUtility.save_image(pic_with_contours, Path(self.output_contour_debug_path), self.jpg_quality)

    def extract_and_save_found_image(self, contour):
        """From given contour, create a rectangle, extract that rectangle from the scanned image and save it."""

        x, y, w, h = cv2.boundingRect(contour)

        img_cropped = self.img_original[y : y + h, x : x + w]  # The found cropped image.

        SharedUtility.save_image(img_cropped, Path(str(self.output_image_path).replace(".jpg", f"_cr_{self.found_images}.jpg")), self.jpg_quality)

        self.found_images += 1


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

    files = SharedUtility.generate_file_list(path=Path(parent_path_images) / Path(input_path))

    if len(files) == 0:
        raise ValueError(f"No image files found in {input_path}\n Exiting.")

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
    test.split_scanned_image()


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
