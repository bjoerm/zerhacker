from ast import Tuple
from pathlib import Path


class ImageParent:
    """Parent class that contains basic operations for dealing with images."""

    def __init__(self, path_input: Path, folder_input: Path, folder_output: Path):
        self.path_input = path_input
        self.path_output_stem, self.path_output_file_extension = self.generate_output_paths(path_input=self.path_input, folder_input=folder_input, folder_output=folder_output)

        self.image_untouched = self.read_image()

    @staticmethod
    def generate_output_paths(path_input: Path, folder_input: Path, folder_output: Path) -> tuple[Path, str]:
        """Generating the parts needed for constructing the output path. This returns the stem and the extension so that possible suffixes can be added to the output stem."""

        path_output = Path(str(path_input).replace(str(folder_input), str(folder_output)))
        path_output_stem = path_output.parent / path_output.stem
        path_output_file_extension = path_output.suffix

        return path_output_stem, path_output_file_extension

    def read_image(self):
        # TODO
        pass


if __name__ == "__main__":

    ImageParent(path_input=Path("test_input_folder/a/abc.txt"), folder_input=Path("test_input_folder/"), folder_output=Path("test_output_folder/"))

    print("End of script reached.")
