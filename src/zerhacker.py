# TODO Check https://github.com/Claytorpedo/scan-cropper/blob/master/scan_cropper.py

# TODO Add documentation to the methods in finecut.
# TODO Low prio: Splitter and finecut don't use the same THRESHOLD system. One uses cv.THRESH_BINARY while the other uses THRESH_BINARY_INV. Change it so that both use the same metric. Splitter should be easier to change w.r.t. this.

import toml

from environment import Environment
from finecut import FineCut
from shared_utils import SharedUtility
from splitter import start_splitting


def main():
    cfg = toml.load("config.toml", _dict=dict)

    Environment.initiate(parent_path_images=cfg.get("parent_path_images"), path_rough_cut=cfg.get("path_rough_cut"), path_fine_cut=cfg.get("path_fine_cut"))

    cfg["num_threads"] = SharedUtility.get_available_threads()

    start_splitting(
        parent_path_images=cfg.get("parent_path_images"),
        path_input=cfg.get("path_untouched_scans"),
        path_output=cfg.get("path_rough_cut"),
        min_pixel_ratio=cfg.get("min_pixel_ratio"),
        detection_threshold=cfg.get("detection_threshold"),
        num_threads=cfg.get("num_threads"),
    )

    FineCut.main(
        parent_path_images=cfg.get("parent_path_images"),
        input_path=cfg.get("path_rough_cut"),
        output_path=cfg.get("path_fine_cut"),
        thresh=cfg.get("detection_threshold_finecut"),
        extra_crop=cfg.get("extra_crop"),
        num_threads=cfg.get("num_threads"),
    )


if __name__ == "__main__":
    main()

    print("\n[Status] Script finished")
