from alfa.data_handling import GesturesDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch


class NeuralNetwork(nn.Module):
    def __init__(self, layers: tuple, batch_size: int = 32, epochs: int = 1, num_classes: int = 18,
                 dir_path: str = None, transform: transforms = None):
        super().__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.dir_path = dir_path

        dataset = GesturesDataset(dir_path=dir_path)

        self.fc1 = nn.Linear(300, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], layers[2])
        self.fc4 = nn.Linear(layers[2], num_classes)

        self.relu = nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(self.device)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        pass

    def predict(self, x):
        """

        :param x:
        :return:
        """
        pass

    def loss(self, x, y):
        pass

    def accuracy(self, x, y):
        pass

