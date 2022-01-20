from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import tqdm

from shared_utils import SharedUtility


class Splitter:
    """This utility class detects and extracts single images from a big scanned image."""

    @classmethod
    def main(
        cls,
        parent_path_images: str,
        input_path: str,
        output_path: str,
        min_pixels: int,
        detection_threshold: int,
        num_threads: int,
    ):

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
                list(
                    tqdm.tqdm(
                        p.imap(cls.split_scanned_image, params), total=len(params)
                    )
                )

        print("\n[Status] Finished Splitter.")

    @classmethod
    def split_scanned_image(cls, params: dict):
        """Detects and extracts single images from a big scanned image."""

        input_image_path = params.get("input_image_path")
        input_path = params.get("input_path")
        output_path = params.get("output_path")
        min_pixels = params.get("min_pixels")
        detection_threshold = params.get("detection_threshold")

        original = cv2.imdecode(
            np.fromfile(input_image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )  # cv2.imread does as of 2021-04 not work for German Umlaute and similar characters. From: https://stackoverflow.com/a/57872297

        thresh = cls.prepare_image(original, detection_threshold)
        cnts = cls.find_contours(thresh)

        # def extract_contours(
        #     image, cnts, min_pixels, input_image_path, input_path, output_path
        # ):
        #     pass

        # Iterate through contours and filter for cropped
        image_number = 0
        too_small_contour_count = 0

        found_pictures = original.copy()

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)

            if w < min_pixels or h < min_pixels:
                too_small_contour_count += 1
                continue  # If the detected rectangle is smaller than x min_pixels, then skip this rectangle.

            # Mark on the big scanned image all found single images.
            found_pictures = cv2.rectangle(
                img=found_pictures,
                pt1=(x, y),
                pt2=(x + w, y + h),
                color=(36, 255, 12),
                thickness=25,
            )

            cropped = original[y : y + h, x : x + w]  # The cropped image.

            # Ensure that (sub) folders for the respective album exist. (Is required if there are folders in the input.)
            output_image_path = str(input_image_path).replace(input_path, output_path)
            Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)

            # Saving image. imwrite does not work with German Umlaute and other special characters. Thus, the following solution.
            # Encode the im_resize into the im_buf_cropped, which is a one-dimensional ndarray (from https://jdhao.github.io/2019/09/11/opencv_unicode_image_path/#write-images-with-unicode-paths)
            is_success, im_buf_cropped = cv2.imencode(
                ".jpg", cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )

            if is_success is True:
                im_buf_cropped.tofile(
                    output_image_path.replace(".jpg", f"_cr_{image_number}.jpg")
                )

            else:
                print("WARNING File could not be read.")

            image_number += 1

        # Mark on the big scanned image all found single images.
        cv2.imwrite(
            filename=str(input_image_path)
            .replace(input_path, output_path)
            .replace(".jpg", "__DEBUG.jpg"),
            img=found_pictures,
        )

        # When no contours are found or only too small contours are detected, copy the image to a special folder in the output folder.
        if len(cnts) == 0 or image_number == too_small_contour_count:
            (
                Path(str(input_image_path).replace(input_path, output_path)).parent
                / Path("no_crop_done")
            ).mkdir(parents=True, exist_ok=True)

            # Saving image. imwrite does not work with German Umlaute and other special characters. Thus, the following solution.
            # Encode the im_resize into the im_buf_cropped, which is a one-dimensional ndarray (from https://jdhao.github.io/2019/09/11/opencv_unicode_image_path/#write-images-with-unicode-paths)
            is_success, im_buf_cropped = cv2.imencode(
                ".jpg", original, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )

            if is_success is True:
                im_buf_cropped.tofile(
                    str(
                        Path(
                            str(input_image_path).replace(input_path, output_path)
                        ).parent
                        / Path("no_crop_done")
                        / Path(input_image_path).name
                    )
                )
            else:
                print("WARNING File could not be read.")

        return

    @staticmethod
    def prepare_image(image, detection_threshold):
        gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(src=gray, ksize=(3, 3), sigmaX=0)
        thresh = cv2.threshold(
            src=blurred,
            thresh=detection_threshold,
            maxval=255,
            type=cv2.THRESH_BINARY_INV,  # For values on thresholds, see: https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
        )[1]

        return thresh

    def find_contours(image):
        # Find contours
        cnts = cv2.findContours(
            image=image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts.reverse()  # Reversing the list so the the found contours start at the top left and not at the bottom.

        return cnts


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

    cfg["num_threads"] = 1  # Default: SharedUtility.get_available_threads()

    Splitter.main(
        parent_path_images=cfg.get("parent_path_images"),
        input_path=cfg.get("untouched_scans_path"),
        output_path=cfg.get("rough_cut_path"),
        min_pixels=cfg.get("min_pixels"),
        detection_threshold=cfg.get("detection_threshold"),
        num_threads=cfg.get("num_threads"),
    )
