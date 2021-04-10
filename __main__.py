# TODO Check https://github.com/Claytorpedo/scan-cropper/blob/master/scan_cropper.py

# TODO Multicrop does not work well with subfolders as of now.
# TODO Add multithreading to multicrop.
# TODO multicrop and potentially other parts do not like Umlaute in paths. Fix this.
# TODO Add documentation to the methods in finecut.
# TODO Tidy up the main function in finecut
# TODO Throw away finecut unused elements.
# TODO Low prio: Multicrop and finebut don't use the same THRESHOLD system. One uses cv.THRESH_BINARY while the other uses THRESH_BINARY_INV. Change it so that both use the same metric. multicrop should be easier to change w.r.t. this.


from shared_utils import SharedUtility
import toml

from environment import Environment
from splitter import Splitter
from finecut import FineCut


def main():
    # Load options
    cfg = toml.load("options.toml", _dict=dict)

    Environment.initiate(cfg.get("parent_path_images"), cfg.get("untouched_scans_path"), cfg.get("rough_cut_path"), cfg.get("fine_cut_path"))

    cfg["num_threads"] = SharedUtility.get_available_threads()

    Splitter.main(parent_path_images=cfg.get("parent_path_images"), input_path=cfg.get("untouched_scans_path"), output_path=cfg.get("rough_cut_path"), min_pixels=cfg.get("min_pixels"), detection_threshold=cfg.get("detection_threshold"), num_threads=cfg.get("num_threads"))

    FineCut.main(parent_path_images=cfg.get("parent_path_images"), input_path=cfg.get("rough_cut_path"), output_path=cfg.get("fine_cut_path"), thresh=cfg.get("detection_threshold_finecut"), extra_crop=cfg.get("extra_crop"), num_threads=cfg.get("num_threads"))


if __name__ == "__main__":
    main()

    print("Script finished")
