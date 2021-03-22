# TODO Check https://github.com/Claytorpedo/scan-cropper/blob/master/scan_cropper.py

import toml

from environment import Environment


def main():
    # Load options
    cfg = toml.load("options.toml", _dict=dict)

    Environment.initiate(parent_path_images=cfg.get("parent_path_images"), folders=cfg.get("folders"))

if __name__ == "__main__":
    main()

    print("Script finished")
