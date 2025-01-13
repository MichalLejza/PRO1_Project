from data_handling import *
from torchvision import transforms
from Model import NeuralNetwork


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    model = NeuralNetwork(layers=(300, 200, 100),
                          dir_path='C:\\Users\\Michał\\PycharmProjects\\PRO_1_Project\\Data\\grammatical_facial_expression',
                          transform=transform)
