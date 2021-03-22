from pathlib import Path

# TODO Maybe add a check for supported file types.


class Environment:
    """ TODO """

    @classmethod
    def initiate(cls, parent_path_images: str, folders: dict):
        """ Setting up the environment os that this script can do its work. """

        cls._ensure_existance_of_required_folders(parent_path_images=parent_path_images, folders=folders)


    @staticmethod
    def _ensure_existance_of_required_folders(parent_path_images: str, folders: dict):
        """ Ensure that all required folders exist (and create them if not). """

        for i in folders.values():
            (Path(parent_path_images) / Path(i)).mkdir(parents=True, exist_ok=True)  # TODO https://stackoverflow.com/questions/17431638/get-typeerror-dict-values-object-does-not-support-indexing-when-using-python
