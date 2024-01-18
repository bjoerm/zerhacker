from pathlib import Path

from image_parent import ImageParent


class ScannedAlbumPage(ImageParent):
    def split(self):
        pass


if __name__ == "__main__":
    ScannedAlbumPage(img_path_input=Path("input/01 - T/doc10074220210228113627_001.jpg"), folder_input=Path("input/"), folder_output=Path("output/1_splitter/"))

    print("End of script reached.")
