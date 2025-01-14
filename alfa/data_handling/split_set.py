from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .dataset import GesturesDataset


class SplitSet(Dataset):
    def __init__(self, gestureDataset: GesturesDataset, train: bool = True, test: bool = False,
                 transform: transforms = None, target_transform: transforms = None):
        self.dataset = gestureDataset
        if train:
            self.dataset = gestureDataset.train_data
        elif test:
            self.dataset = gestureDataset.test_data
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
        pass

    def _split_data(self) -> tuple:
        """

        :return:
        """
        pass

    def get_data_loader(self) ->  DataLoader:
        """

        :return:
        """
        pass

    def print_dataset_info(self) -> None:
        """

        :return: None
        """
        pass