from pathlib import Path
import os

# TODO Maybe add a check for supported file types.


class Environment:
    """ TODO """

    @classmethod
    def initiate(cls, parent_path_images: str, *folders: tuple):
        """ Setting up the environment os that this script can do its work. """

        cls._ensure_existance_of_required_folders(parent_path_images=parent_path_images, folders=folders)

    @staticmethod
    def _ensure_existance_of_required_folders(parent_path_images: str, folders: tuple):
        """ Ensure that all required folders exist (and create them if not). """

        for i in folders:
            (Path(parent_path_images) / Path(i)).mkdir(parents=True, exist_ok=True)  # TODO https://stackoverflow.com/questions/17431638/get-typeerror-dict-values-object-does-not-support-indexing-when-using-python

    @staticmethod
    def get_available_threads():

        try:
            num_threads = os.cpu_count()
        except AttributeError:
            try:
                num_threads = multiprocessing.cpu_count()
            except NotImplementedError:
                print("Automatic thread detection didn't work. Defaulting to 1 thread only.")
                num_threads = 1

        print(f"Using {num_threads} threads.")

        return num_threads
