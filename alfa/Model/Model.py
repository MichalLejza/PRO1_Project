from torch import optim
from tqdm import tqdm
from alfa.data_handling import GesturesDataset, SplitSet
from torchvision import transforms
import torch.nn as nn
import torch
from .Neural_Network import NeuralNetwork


class Model(nn.Module):
    def __init__(self, layers: tuple, batch_size: int = 32, epochs: int = 10, num_classes: int = 18,
                 dir_path: str = None):
        super().__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.dir_path = dir_path

        self.model = NeuralNetwork(layers, num_classes)

        dataset = GesturesDataset(dir_path=dir_path)
        dataset.print_dataset_info()

        self.train_data = SplitSet(gestureDataset=dataset, train=True)
        self.test_data = SplitSet(gestureDataset=dataset, test=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_model(self):
        self.model.train()
        train_loader = self.train_data.get_data_loader()
        print(next(iter(train_loader))[0])
        for epoch in range(self.epochs):
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.epochs}: '):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f'  Loss: {running_loss / len(train_loader):.4f}')
            self.test_model()
        print('Training completed.')

    def test_model(self):
        self.model.eval()
        test_loader = self.test_data.get_data_loader()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")
