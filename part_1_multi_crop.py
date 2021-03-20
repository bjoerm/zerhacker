# TODO Check: https://github.com/z80z80z80/autocrop and maybe https://github.com/msaavedra/autocrop


# Original source from https://github.com/numpy/numpy/issues/17726


import cv2
import glob
import pathlib

MIN_PIXELS = 300

pathlib.Path("output/multi_crop").mkdir(parents=True, exist_ok=True) # Ensuring that the output folder exists.

files_list = glob.glob(
    "input/multi_crop/**/*.jpg"  # glob itself seems to not care about whether file extension contain capital letters. Anyways, [mM][pP]3 is a bit safer than just mp3. This does not perceive .mp3a as .mp3. TODO Check whether it has problems with folders that contains the .mp3 somewhere.
    , recursive=True # Recursive = True in combination with ** in the path will lead to a search in every folder and subfolder and subsubfolder and ...
    )


for i in files_list:
    image = cv2.imread(i)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # thresh = cv2.threshold(blurred, 230,255,cv2.THRESH_BINARY_INV)[1] # Original values.
    thresh = cv2.threshold(blurred, 190, 255, cv2.THRESH_BINARY_INV)[1] # The first value after "blurred" is the threshold. I think it can be set between 0 and 255. The higher the higher the threshold. https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts.reverse() # Reversing the list so the the found pictures start at the top left and not at the bottom.

    # Iterate thorugh contours and filter for ROI
    image_number = 0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)

        if w < MIN_PIXELS or h < MIN_PIXELS:
            continue # If the detected rectangle is smaller than MIN_PIXELS x MIN_PIXELS, then skip this rectangle.

        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)

        ROI = original[y:y+h, x:x+w] # The cropped image.

        cv2.imwrite(str(i).replace("input", "output").replace(".jpg", "") + "_cr_{}.jpg".format(image_number), ROI)

        image_number += 1
