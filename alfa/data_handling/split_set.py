import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from .dataset import GesturesDataset


class SplitSet(Dataset):
    def __init__(self, gestureDataset: GesturesDataset, train: bool = False, test: bool = False,
                 transform: transforms = None, target_transform: transforms = None):
        if train:
            self.dataset = gestureDataset.train_data
        elif test:
            self.dataset = gestureDataset.test_data
        self.categories_map = gestureDataset.categories_map
        self.filter_dataset()
        self.data, self.target = self._split_data()
        self.transform = transform
        self.target_transform = target_transform

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

    def filter_dataset(self) -> None:
        """

        :return:
        """
        for row in self.dataset:
            if row[-2] == 0:
                self.dataset.remove(row)

    def _split_data(self) -> tuple:
        """

        :return:
        """
        data: list[list[float]] = []
        targets: list[int] = []

        for row in self.dataset:
            data.append(row[1:-2])
            targets.append(self.categories_map[row[-1]])

        return data, targets

    def get_data_loader(self) ->  DataLoader:
        """

        :return:
        """
        tensor_data = torch.Tensor(self.data)
        tensor_target = torch.Tensor(self.target)
        dataset = TensorDataset(tensor_data, tensor_target)
        return DataLoader(dataset, batch_size=32, shuffle=True)



    def print_dataset_info(self) -> None:
        """

        :return: None
        """
        print('Zbiór Danych: Gramatical facial Expression')
        print(f'Wielkość zbioru danych: {len(self.data)}')
        print(f'Ilość punktów: {len(self.data[0])}')
        print(f'Ilość kategorii w zbiorze: {len(self.categories_map.items())}')