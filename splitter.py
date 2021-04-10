# TODO Add parallel execution. See the autocrop part.
# TODO Add status messages or progress bar.

import cv2
from pathlib import Path
from shared_utils import SharedUtility
from multiprocessing import Pool


class Splitter:
    """ This utility class detects and extracts single images from a big scanned image. """

    @classmethod
    def main(cls, parent_path_images: str, input_path: str, output_path: str, min_pixels: int, detection_threshold: int, num_threads: int):
        """ TODO """

        files = SharedUtility.generate_file_list(path=Path(parent_path_images) / Path(input_path))

        if len(files) == 0:
            print(f"No image files found in {input_path}\n Exiting.")  # TODO Add stop of the programm.

        else:
            # Creating list of dictionaries for parallel processing.
            params = []

            for f in files:
                params.append({
                    "input_image_path": f
                    , "input_path": input_path
                    , "output_path": output_path
                    , "min_pixels": min_pixels
                    , "detection_threshold": detection_threshold
                    })

            # Parallel slitting of the scanned images.
            with Pool(num_threads) as p:
                p.map(cls._split_scanned_image, params)

        print("\n[Status] Finished Splitter.")

    @staticmethod
    def _split_scanned_image(params: dict):
        """ Original source from https://github.com/numpy/numpy/issues/17726 """

        input_image_path = params["input_image_path"]
        input_path = params["input_path"]
        output_path = params["output_path"]
        min_pixels = params["min_pixels"]
        detection_threshold = params["detection_threshold"]

        image = cv2.imread(filename=input_image_path)
        original = image.copy()
        gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(src=gray, ksize=(3, 3), sigmaX=0)
        thresh = cv2.threshold(src=blurred, thresh=detection_threshold, maxval=255, type=cv2.THRESH_BINARY_INV)[1]  # For values on thresholds, see: https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

        # Find contours
        cnts = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts.reverse()  # Reversing the list so the the found pictures start at the top left and not at the bottom.

        # Iterate through contours and filter for cropped
        image_number = 0
        too_small_contour_count = 0

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)

            if w < min_pixels or h < min_pixels:
                too_small_contour_count += 1
                continue  # If the detected rectangle is smaller than x min_pixels, then skip this rectangle.

            # # Debugging:
            # # If you want to see the found rectangle contour on the original image.
            # test = cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(36, 255, 12), thickness=5)
            # cv2.imwrite(filename=str(input_image_path).replace(input_path, output_path).replace(".jpg", "DEBUGTEST.jpg"), img=test)

            cropped = original[y: y + h, x: x + w]  # The cropped image.

            # Create folders, if not exists. (Is required if there are folders in the input.)
            output_image_path = str(input_image_path).replace(input_path, output_path)
            Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_image_path.replace(".jpg", f"_cr_{image_number}.jpg"), cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            image_number += 1

        # When no contours are found or only too small contours are detected, copy the image to a special folder in the output folder.
        if len(cnts) == 0 or image_number == too_small_contour_count:
            (Path(str(input_image_path).replace(input_path, output_path)).parent / Path("no_crop_done")).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(input_image_path).replace(input_path, output_path + "/no_crop_done/"), original, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        return
