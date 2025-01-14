from sklearn.model_selection import train_test_split
from .file_handler import *
from pandas import DataFrame


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
        self.dataset: list[list] = self._read_files()
        self.train_data, self.test_data = train_test_split(self.dataset, test_size=0.2, random_state=42)
        self.categories_map: dict = self._get_categories_map()

    def _read_files(self) -> list[list]:
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

    def get_data_frame(self) -> DataFrame:
        """
        Metoda zwraca DataFrame z danymi z listy.

        :return: Dataframe z danymi z plików
        """
        return DataFrame(self.dataset)

    def print_dataset_info(self) -> None:
        """
        Metoda wypisuje informacje o zbiorze danych:
        Ilość punktów, ilość danych w zbiorze i ilość kategorii w zbiorze.

        :return: None
        """
        print('Zbiór Danych: Gramatical facial Expression')
        print(f'Wielkość zbioru danych: {len(self.dataset)}')
        print(f'Ilość punktów: {len(self.dataset[0][1:-2])}')
        print(f'Ilość kategorii w zbiorze: {len(self.categories_map.items())}')

    def _get_categories_map(self) -> dict:
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
