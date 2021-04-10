import glob
from pathlib import Path
import os


class SharedUtility:
    """ This class contains shared utility methods. """

    @staticmethod
    def generate_file_list(path: str) -> list:
        """ Generate a list of all images in the input path. """

        file_list_gen = Path(path)
        file_list = [str(f.parent / f.name) for f in file_list_gen.rglob("*.jpg")]  # TODO Deal with .jpeg, .JPG, ... # rglob is recursive.

        return file_list

    @staticmethod
    def get_available_threads() -> int:
        """ Get number of available CPU threads that can be used for parallel processing. """

        try:
            num_threads = os.cpu_count()  # This is not 100% the best approach but os.sched_getaffinity does not work on Windows.

        except NotImplementedError:
            print("Automatic thread detection didn't work. Defaulting to 1 thread only.")
            num_threads = 1

        print(f"Using {num_threads} threads.")

        return num_threads
