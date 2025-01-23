from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt

from .dataset import GesturesDataset
from .prepare_data import filter_dataset, split_data, to_pytorch_tensor, standarise_data


class SplitSet(Dataset):
    def __init__(self, gestureDataset: GesturesDataset, train: bool = False, test: bool = False, taransform = None,
                 batch_size: int = 32):
        """
        Klasa przechowująca gotowy zbior danych do trenowania i testowania
        :param gestureDataset: Instancja klasy GesturesDataset
        :param train: czy zbior treningowy
        :param test: czy zbior testowy
        :param taransform: funkcja transformacji danych
        :param batch_size: rozmiar batcha
        """
        if train:
            self.dataset = gestureDataset.train_data
        elif test:
            self.dataset = gestureDataset.test_data
        self.categories_map = gestureDataset.categories_map
        self.transform = taransform
        self.batch_size = batch_size
        self.data, self.target = self.__prepare_data(self.dataset, self.categories_map)

    def __len__(self) -> int:
        """
        Metoda zwracająca wielkość zbioru
        :return: Wielkość zbioru treningowego/testowego
        """
        return len(self.dataset)

    def __getitem__(self, item) -> tuple:
        """
        Metoda zwracająca element zbioru o danym indexie
        :param item: index elementu
        :return: element i target
        """
        data = self.data[item]
        target = self.target[item]
        if self.transform:
            data = self.transform(data)
        return data, target

    def get_gesture(self, index: int) -> str:
        """
        Metoda zwracająca nazwe gestu na podstawie indexu od 0 do 17
        :param index: index zbioru
        :return: Nazwa gestu
        """
        for category, value in self.categories_map.items():
            if value == index:
                return category
        # to do exception

    @staticmethod
    def __prepare_data(data: list[list[float]], categories: dict) -> tuple:
        """
        Metoda do przygotowania danych do trenowania i testowania
        :return: Zbiór danych w postaci tensora Pytorch i target
        """
        dataset = filter_dataset(data)
        dataset, targets = split_data(dataset, categories)
        dataset, targets = to_pytorch_tensor(dataset, targets)
        dataset = standarise_data(dataset)
        return dataset, targets

    def get_data_loader(self) ->  DataLoader:
        """
        Metoda zwraca dataloader
        :return: dataloader
        """
        dataset = TensorDataset(self.data, self.target)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def print_dataset_info(self) -> None:
        """
        Metoda do wypisania informacji o zbiorze danych
        :return: None
        """
        print('Zbiór Danych: Gramatical facial Expression')
        print(f'Wielkość zbioru danych: {len(self.data)}')
        print(f'Ilość punktów: {len(self.data[0])}')
        print(f'Ilość kategorii w zbiorze: {len(self.categories_map.items())}')

    def show_class_distribution(self) -> None:
        """
        Metoda do wyswietlenia rozkladu ilosciowego klas
        :return: None
        """
        unique_classes, counts = self.target.unique(return_counts=True)
        class_labels = [self.get_gesture(index) for index in unique_classes]
        plt.figure(figsize=(22, 8))
        bars = plt.bar(class_labels, counts.tolist(), color='skyblue')
        for bar, count in zip(bars, counts.tolist()):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(count), ha='center', fontsize=12)
        plt.title('Rozkład ilościowy klas', fontsize=16)
        plt.xlabel('Klasy', fontsize=14)
        plt.ylabel('Liczba wystąpień', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()