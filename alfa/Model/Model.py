import torch.nn as nn
from torch import optim, device, cuda, max, no_grad

from alfa.data_handling import GesturesDataset, SplitSet
from .Neural_Network import NeuralNetwork


class Model:
    def __init__(self, layers: tuple, batch_size: int = 32, num_classes: int = 18, dir_path: str = None):
        """

        :param layers:
        :param batch_size:
        :param num_classes:
        :param dir_path:
        """
        super().__init__()
        self.model = NeuralNetwork(layers, num_classes)

        dataset = GesturesDataset(dir_path=dir_path)
        self.train_data = SplitSet(gestureDataset=dataset, train=True, batch_size=batch_size)
        self.test_data = SplitSet(gestureDataset=dataset, test=True, batch_size=batch_size)

        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_model(self, epochs: int = 10):
        """

        :param epochs:
        :return:
        """
        self.model.train()
        train_loader = self.train_data.get_data_loader()

        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            print(f'Epoch {epoch + 1}: Loss: {running_loss / len(train_loader):.4f} ', end=' ')
            self.test_model()
        print('Training completed.')

    def test_model(self):
        """

        :return:
        """
        self.model.eval()
        test_loader = self.test_data.get_data_loader()
        correct, total = 0, 0

        with no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")
