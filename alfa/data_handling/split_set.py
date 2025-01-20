from torch.utils.data import Dataset, DataLoader, TensorDataset

from .dataset import GesturesDataset
from .prepare_data import filter_dataset, split_data, to_pytorch_tensor, standarise


class SplitSet(Dataset):
    def __init__(self, gestureDataset: GesturesDataset, train: bool = False, test: bool = False, taransform = None):
        if train:
            self.dataset = gestureDataset.train_data
        elif test:
            self.dataset = gestureDataset.test_data
        self.categories_map = gestureDataset.categories_map
        self.transform = taransform
        self.data, self.target = self.__prepare_data(self.dataset, self.categories_map)

    def __len__(self) -> int:
        """

        :return: Wielkość zbioru treningowego/testowego
        """
        return len(self.dataset)

    def __getitem__(self, item) -> tuple:
        """

        :param item:
        :return:
        """
        data = self.data[item]
        target = self.target[item]
        if self.transform:
            data = self.transform(data)
        return data, target

    def get_gesture(self, index: int) -> str:
        """

        :param index:
        :return:
        """
        for category, value in self.categories_map.items():
            if value == index:
                return category
        # to do exception

    @staticmethod
    def __prepare_data(data: list[list[float]], categories: dict) -> tuple:
        """

        :return:
        """
        dataset = filter_dataset(data)
        dataset, targets = split_data(dataset, categories)
        dataset, targets = to_pytorch_tensor(dataset, targets)
        dataset = standarise(dataset)
        return dataset, targets

    def get_data_loader(self) ->  DataLoader:
        """

        :return:
        """
        dataset = TensorDataset(self.data, self.target)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def print_dataset_info(self) -> None:
        """

        :return: None
        """
        print('Zbiór Danych: Gramatical facial Expression')
        print(f'Wielkość zbioru danych: {len(self.data)}')
        print(f'Ilość punktów: {len(self.data[0])}')
        print(f'Ilość kategorii w zbiorze: {len(self.categories_map.items())}')