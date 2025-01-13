from sklearn.model_selection import train_test_split
from .file_handler import *
from pandas import DataFrame


class GesturesDataset:
    def __init__(self, dir_path: str):
        self.dir_path: str = dir_path
        self.dataset: list[list] = self._read_files()
        self.train_data, self.test_data = train_test_split(self.dataset, test_size=0.2, random_state=42)
        self.categories_map: dict = self._get_categories_map()

    def _read_files(self) -> list[list]:
        """

        :return:
        """
        paths: list[str] = get_datasets_names_in_directory(self.dir_path)
        splitted: dir = split_datasets_names_in_directory(paths)
        points: list[list] = get_face_points(splitted)
        return points

    def get_data_frame(self) -> DataFrame:
        """

        :return:
        """
        return DataFrame(self.dataset)

    def print_dataset_info(self) -> None:
        """

        :return:
        """
        pass

    def _get_categories_map(self) -> dict:
        """

        :return:
        """
        categories_map: dict = {}
        index = 0
        for category in self.dataset:
            if category[1] not in categories_map:
                categories_map[category[1]] = index
                index += 1
        return categories_map
