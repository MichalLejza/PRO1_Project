from sklearn.model_selection import train_test_split
from .file_handler import *


class GesturesDataset:
    def __init__(self, dir_path: str):
        """
        Klasa do otworzenia danych i przechowania danych w tablicy
        aby potem można było je przekazać do klasy SplitSet.
        Przechowuje surowe dane, ścieżki do plików i można wypisać informacje
        o całym zbiorze danych

        :param dir_path: Ścieżka do folderu ../grammatical_facial_expression
        """
        self.dir_path: str = dir_path
        self.dataset: list[list] = self.__read_files()
        self.train_data, self.test_data = train_test_split(self.dataset, test_size=0.2, random_state=42)
        self.categories_map: dict = self.__get_categories_map()

    def __read_files(self) -> list[list]:
        """
        Metoda wykorzystuje metody z pliku file_handler do otworzenia
        i odpowiedniego podzielenia plików i zawartości. Celem jest zwrócenie listy
        zawierającej listy z danymi: punktami i targetem i nazwą wyrazu twarzy.

        :return: Lista list z danymi: punktami i targetem i nazwą wyrazu twarzy
        """
        paths: list[str] = get_datasets_names_in_directory(self.dir_path)
        splitted: dir = split_datasets_names_in_directory(paths)
        points: list[list] = get_face_points(splitted)
        return points

    def __get_categories_map(self) -> dict:
        """
        Metoda zwraca słownik z kategoriami w zbiorze danych. Każda kategoria zawiera
        unikalny numer od 0 do 17.

        :return: Mapa z kategoriami i odpowiednimi numerami
        """
        categories_map: dict = {}
        index = 0
        for category in self.dataset:
            if category[-1] not in categories_map:
                categories_map[category[-1]] = index
                index += 1
        return categories_map
