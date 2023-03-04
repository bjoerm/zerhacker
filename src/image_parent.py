from pathlib import Path


class ImageParent:
    """Parent class that contains basic operations."""

    def __init__(self, path_input: Path, folder_input: Path, folder_output: Path):
        self.path_input = path_input
        self.generate_output_paths()

        self.image_untouched = self.read_image()

    def generate_output_paths(self):
        self.path_output_stem = ""
        self.path_output_file_extension = ""

    def read_image(self):
        pass


if __name__ == "__main__":

    ImageParent(path_input=Path("abc.txt"), folder_output=Path("test_output_folder/"))

    print("End of script reached.")
