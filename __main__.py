# TODO Check https://github.com/Claytorpedo/scan-cropper/blob/master/scan_cropper.py

import toml

from environment import Environment
from multicrop import MultiCrop
from finecut import FineCut

def main():
    # Load options
    cfg = toml.load("options.toml", _dict=dict)


    Environment.initiate(cfg.get("parent_path_images"), cfg.get("untouched_scans_path"), cfg.get("rough_cut_path"), cfg.get("fine_cut_path"))

    # TODO Nice to have: A class that rotates pictures automatically (e.g. if they are scanned upside down).

    MultiCrop.main(parent_path_images=cfg.get("parent_path_images"), input_path=cfg.get("untouched_scans_path"), output_path=cfg.get("rough_cut_path"), min_pixels=cfg.get("min_pixels"), detection_threshold=cfg.get("detection_threshold"))

    FineCut.main(parent_path_images=cfg.get("parent_path_images"), in_path=cfg.get("rough_cut_path"), out_path=cfg.get("fine_cut_path"), thresh=200, crop=5, num_threads=4)

if __name__ == "__main__":
    main()

    print("Script finished")
