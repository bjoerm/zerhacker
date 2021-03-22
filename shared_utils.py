import glob
from pathlib import Path


class SharedUtility:
    """ This class contains shared utility methods. """

    @staticmethod
    def generate_file_list(path: str) -> list:
        """ Generate a list of all images in the input path. """

        file_list_gen = Path(path)
        file_list = [str(f.parent / f.name) for f in file_list_gen.rglob("*.jpg")]  # TODO Deal with .jpeg, .JPG, ... # rglob is recursive.

        return file_list
