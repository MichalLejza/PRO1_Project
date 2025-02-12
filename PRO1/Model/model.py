import torch
import torch.nn as nn
from torch import optim, device, cuda, max, no_grad
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from PRO1.data_handling import GesturesDataset, SplitSet
from .neural_network import NeuralNetwork


class Model:
    def __init__(self, layers: tuple, batch_size: int = 32, num_classes: int = 18, dir_path: str = None):
        """
        Klasa do rozpoznawania emocji z plików w folderze ../grammatical_facial_expression
        Zawiera model sieci neuronowych i możliwość uczenia, testowania, zapisywania modelu
        :param layers: Krotka z liczba neuronów w każdej warstwie
        :param batch_size: Rozmiar batcha
        :param num_classes: Liczba klas do rozpoznania
        :param dir_path: Sciezka do folderu ../grammatical_facial_expression
        """
        super().__init__()
        self.model = NeuralNetwork(layers, num_classes)

        dataset = GesturesDataset(dir_path=dir_path)
        print(dataset.categories_map)
        print(len(dataset.dataset))
        self.train_data = SplitSet(gestureDataset=dataset, train=True, batch_size=batch_size)
        print(self.train_data.data.shape)
        self.test_data = SplitSet(gestureDataset=dataset, test=True, batch_size=batch_size)
        print(self.test_data.data.shape)

        self.train_data.show_class_distribution()
        self.test_data.show_class_distribution()

        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_history = []
        self.acc_history = []

    def train_model(self, epochs: int = 10) -> None:
        """
        Metoda do uczenia modelu
        :param epochs: Liczba epok
        :return: None
        """
        self.model.train()
        train_loader = self.train_data.get_data_loader()
        print('Training started.')

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

            self.loss_history.append(running_loss / len(train_loader))
            print(f'Epoch {epoch + 1}: Loss: {running_loss / len(train_loader):.4f} ', end=' ')
            self.test_model()

        print('Training completed.')

    def test_model(self) -> None:
        """
        Metoda do testowania modelu
        :return: None
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
        self.acc_history.append(accuracy)
        print(f"Accuracy: {accuracy:.2f}%")

    def save_model(self, path: str) -> None:
        """
        Metoda do zapisywania modelu
        :param path: Sciezka do zapisu
        :return: None
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        """
        Metoda do wczytywania modelu
        :param path: Sciezka do wczytywania
        :return: None
        """
        self.model.load_state_dict(torch.load(path))

    def display_loss(self) -> None:
        """
        Metoda do wyswietlenia historii straty
        :return: None
        """
        epochs = range(1, len(self.loss_history) + 1)  # Numery epok

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.loss_history, 'b', marker='o', label='loss')
        plt.title('Historia błędów modelu', fontsize=16)
        plt.xlabel('Epoka', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.xticks(epochs)
        plt.ylim(0, 2)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    def display_accuracy(self) -> None:
        """
        Metoda do wyswietlenia dokladnosci
        :return: None
        """
        epochs = range(1, len(self.acc_history) + 1)  # Numery epok

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.acc_history, 'b-', marker='o', label='Dokładność')
        plt.title('Historia dokładności modelu', fontsize=16)
        plt.xlabel('Epoka', fontsize=14)
        plt.ylabel('Dokładność', fontsize=14)
        plt.ylim(0, 100)  # Zakres osi Y od 0 do 1 (dla dokładności w ułamkach)
        plt.xticks(epochs)  # Wyświetl epoki jako wartości na osi X
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    def display_conf_matrix(self) -> None:
        """
        Metoda do wyswietlenia macierzy pomyłek
        :return: None
        """
        self.model.eval()
        test_loader = self.test_data.get_data_loader()
        true_labels = []
        predicted_labels = []
        with no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = max(outputs, 1)
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()


