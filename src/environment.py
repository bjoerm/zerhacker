import shutil
from pathlib import Path

# TODO Merge this with the shared_utils class?


class Environment:
    @classmethod
    def initiate(cls, parent_path_images: str, path_rough_cut: str, path_fine_cut: str):
        """Setting up the environment so that this script can do its work."""

        for i in [path_rough_cut, path_fine_cut]:
            shutil.rmtree(f"{parent_path_images}/{i}", ignore_errors=True)

        cls._ensure_existance_of_required_folders(parent_path_images, path_rough_cut, path_fine_cut)

    @staticmethod
    def _ensure_existance_of_required_folders(parent_path_images: str, *folders: tuple):
        """Ensure that all required folders exist (and create them if not)."""

        for i in folders:
            (Path(parent_path_images) / Path(i)).mkdir(parents=True, exist_ok=True)
            # TODO https://stackoverflow.com/questions/17431638/get-typeerror-dict-values-object-does-not-support-indexing-when-using-python
