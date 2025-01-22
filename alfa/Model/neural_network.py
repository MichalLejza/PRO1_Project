import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, layers: tuple, num_classes: int = 18):
        """

        :param layers: Krotka z liczba neuronów w każdej warstwie
        :param num_classes: Liczba klas do rozpoznania
        """
        super().__init__()
        self.fc1 = nn.Linear(300, layers[0], bias=True)
        self.fc2 = nn.Linear(layers[0], layers[1], bias=True)
        self.fc3 = nn.Linear(layers[1], layers[2], bias=True)
        self.fc4 = nn.Linear(layers[2], num_classes, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        """

        :param x: Dane wejściowe, batch z danymi
        :return: Dane wyjściowe
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x