from pathlib import Path


class Environment:
    """ TODO """

    @classmethod
    def initiate(cls, parent_path_images: str, folders: dict):
        """ TODO """

        cls._ensure_existance_of_required_folders(parent_path_images=parent_path_images, folders=folders)


    @staticmethod
    def _ensure_existance_of_required_folders(parent_path_images: str, folders: dict):
        """ Ensure that all required folders exist (and create them if not). """

        for i in folders.values():
            (Path(parent_path_images) / Path(i)).mkdir(parents=True, exist_ok=True)
