from data_handling import *
from torchvision import transforms
from Model import Model


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    model = Model(layers=(300, 200, 100),
                  dir_path='C:\\Users\\Michał\\PycharmProjects\\PRO_1_Project\\Data\\grammatical_facial_expression',
                  transform=transform)
    model.train_model()
