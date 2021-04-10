# Original source: https://github.com/z80z80z80/autocrop

import cv2
import numpy as np
import os
import glob
from multiprocessing import Pool
import multiprocessing
from pathlib import Path
from shared_utils import SharedUtility


class FineCut:
    """Crop/Rotate images automatically. Images should be single images on white background."""

    @classmethod
    def main(cls, parent_path_images: str, input_path: str, output_path: str, thresh: int, extra_crop: int, num_threads: int):
        """
        Threshold value: Higher values represent less aggressive contour search. If it's chosen too high, a white border will be introduced.

        Extra crop: After crop/rotate often a small white border remains. This removes this. If it cuts off too much of your image, adjust this.

        Number of threads to be used to process the images in parallel. If not provided, the script will try to find the value itself (which doesn't work on Windows or MacOS -> defaults to 1 thread only).
        """

        input_path = Path(parent_path_images) / Path(input_path)
        output_path = Path(parent_path_images) / Path(output_path)

        files = SharedUtility.generate_file_list(path=input_path)

        if len(files) == 0:
            print(f"No image files found in {input_path}\n Exiting.")  # TODO Add stop of the programm.

        else:
            # Creating list of dictionaries for parallel processing.
            params = []

            for f in files:
                params.append({
                    "thresh": thresh
                    , "crop": extra_crop
                    , "filename": f
                    , "output_path": output_path
                    })

            # Parallel fine cut of the scanned images.
            with Pool(num_threads) as p:
                p.map(cls.autocrop, params)

            print("\n[Status] Finished FineCut.")

    @classmethod
    def autocrop(cls, params: dict):
        thresh = params["thresh"]
        crop = params["crop"]
        filename = params["filename"]
        output_path = params["output_path"]

        print(f"Opening: {filename}")
        name = Path(filename).name
        img = cv2.imread(filename)

        # Add white background (in case one side is cropped right already, otherwise script would fail finding contours)
        img = cv2.copyMakeBorder(src=img, top=100, bottom=100, left=100, right=100, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        im_h, im_w = img.shape[:2]
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        res_gray = cv2.resize(src=img, dsize=(int(im_w / 6), int(im_h / 6)), interpolation=cv2.INTER_CUBIC)  # TODO What is this need for? Try to understand before dropping it?
        found, img = cls.cont(img=img, gray=gray, user_thresh=thresh, crop=crop)

        if found:
            print(f"Saving to: {output_path}/{name}")
            try:
                cv2.imwrite(f"{output_path}/{name}", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            except:
                None
            # TODO: This is always writing JPEG, no matter what the input file type was. Nice to have: Change this.

        else:
            # If no contours were found, write input file to "failed" folder
            print(f"Failed finding any contour. Saving original file to {output_path}/failed/{name}")
            if not os.path.exists(f"{output_path}/failed/"):
                os.makedirs(f"{output_path}/failed/")

            with open(filename, "rb") as in_f, open(f"{output_path}/failed/{name}", "wb") as out_f:
                while True:
                    buf = in_f.read(1024**2)
                    if not buf:  # TODO Understand this. When will this be triggered?
                        break
                    else:
                        out_f.write(buf)

    def order_rect(points):
        # idea: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
        # initialize result -> rectangle coordinates (4 corners, 2 coordinates (x,y))
        res = np.zeros((4, 2), dtype=np.float32)

        # top-left corner: smallest sum
        # top-right corner: smallest difference
        # bottom-right corner: largest sum
        # bottom-left corner: largest difference

        s = np.sum(points, axis=1)
        d = np.diff(points, axis=1)

        res[0] = points[np.argmin(s)]
        res[1] = points[np.argmin(d)]
        res[2] = points[np.argmax(s)]
        res[3] = points[np.argmax(d)]

        return res

    @classmethod
    def four_point_transform(cls, img, points):
        # copied from: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
        # obtain a consistent order of the points and unpack them individually
        rect = cls.order_rect(points)
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the maximum distance between the top-right and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([[0, 0],
                        [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1],
                        [0, maxHeight - 1]], dtype=np.float32)

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        # return the warped image
        return warped

    @classmethod
    def cont(cls, img, gray, user_thresh, crop):
        found = False
        loop = False
        old_val = 0  # thresh value from 2 iterations ago
        i = 0  # number of iterations

        im_h, im_w = img.shape[:2]
        while found is False:  # repeat to find the right threshold value for finding a rectangle
            if user_thresh >= 255 or user_thresh == 0 or loop:  # maximum threshold value, minimum threshold value or loop detected (alternating between 2 threshold values without finding borders.
                break  # stop if no borders could be detected

            thresh = cv2.threshold(src=gray, thresh=user_thresh, maxval=255, type=cv2.THRESH_BINARY)[1]
            contours = cv2.findContours(image=thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)[0]
            im_area = im_w * im_h

            for cnt in contours:
                area = cv2.contourArea(contour=cnt)
                if area > (im_area / 100) and area < (im_area / 1.01):
                    epsilon = 0.1 * cv2.arcLength(curve=cnt, closed=True)
                    approx = cv2.approxPolyDP(curve=cnt, epsilon=epsilon, closed=True)

                    if len(approx) == 4:
                        found = True
                    elif len(approx) > 4:
                        user_thresh = user_thresh - 1
                        print(f"Adjust Threshold: {user_thresh}")
                        if user_thresh == old_val + 1:
                            loop = True
                        break
                    elif len(approx) < 4:
                        user_thresh = user_thresh + 5
                        print(f"Adjust Threshold: {user_thresh}")
                        if user_thresh == old_val - 5:
                            loop = True
                        break

                    rect = np.zeros((4, 2), dtype=np.float32)
                    rect[0] = approx[0]
                    rect[1] = approx[1]
                    rect[2] = approx[2]
                    rect[3] = approx[3]

                    dst = cls.four_point_transform(img=img, points=rect)
                    dst_h, dst_w = dst.shape[:2]
                    img = dst[crop: dst_h - crop, crop: dst_w - crop]
                else:
                    if i > 100:
                        # if this happens a lot, increase the threshold, maybe it helps, otherwise just stop
                        user_thresh = user_thresh + 5
                        if user_thresh > 255:
                            break
                        print(f"Adjust Threshold: {user_thresh}")
                        print("WARNING: This seems to be an edge case. If the result isn't satisfying try lowering the threshold using -t")

                        if user_thresh == old_val - 5:
                            loop = True
            i += 1
            if i % 2 == 0:
                old_value = user_thresh  # TODO What is this part for?

        return found, img
