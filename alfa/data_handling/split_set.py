from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .dataset import GesturesDataset


class SplitSet(Dataset):
    def __init__(self, dataset: GesturesDataset, train: bool = True, test: bool = False,
                 transform: transforms = None, target_transform: transforms = None):
        self.dataset = dataset

    def __len__(self) -> int:
        """

        :return:
        """
        pass

    def __getitem__(self, item) -> tuple:
        """

        :param item:
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

        :return:
        """
        pass