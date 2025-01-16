import torch.nn as nn
import torch


class NeuralNetwork(nn.Module):
    def __init__(self, layers: tuple, num_classes: int = 18, *args, **kwargs):
        """

        :param layers:
        :param num_classes:
        """
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(300, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], layers[2])
        self.fc4 = nn.Linear(layers[2], num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x