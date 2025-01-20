from Model import Model


if __name__ == '__main__':
    model = Model(layers=(300, 200, 100),
                  dir_path='C:\\Users\\Michał\\PycharmProjects\\PRO_1_Project\\Data\\grammatical_facial_expression',
                  epochs=30)
    model.train_model()
