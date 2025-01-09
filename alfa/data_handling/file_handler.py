import os
from pandas import DataFrame


def get_datasets_names_in_directory(path: str) -> list[str]:
    """

    :param path:
    :return:
    """
    if not os.path.isdir(path):
        raise NotADirectoryError(f'{path} is not a directory')

    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} does not exist')

    files: list[str] = [(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    datasets: list[str] = [f for f in files if f.endswith('.txt')]

    if not datasets:
        raise FileNotFoundError(f'No dataset files in {path}')

    return datasets


def split_datasets_names_in_directory(paths: list[str]) -> tuple[list[str], list[str]]:
    """

    :param paths:
    :return:
    """
    datapoints: list[str] = []
    targets: list[str] = []

    for path in paths:
        if path.endswith('_datapoints.txt'):
            datapoints.append(path)
        elif path.endswith('_targets.txt'):
            targets.append(path)

    return datapoints, targets


def get_face_points(datapoints: list[str]) -> tuple[list[list[float]], list[str]]:
    """

    :param datapoints:
    :return:
    """
    with open(datapoints[0], 'r') as f:
        columns = f.readline().strip().split(' ')
    f.close()

    data_points: list[list[float]] = []

    for file in datapoints:
        with open(file, 'r') as f:
            f.readline()
            for line in f:
                line = list(map(float, line.strip().split(' ')))
                data_points.append(line)
        f.close()

    return data_points, columns


def get_targets(targets: list[str]) -> list[list[str]]:
    """

    :param targets:
    :return:
    """


if __name__ == '__main__':
    paths = get_datasets_names_in_directory('C:\\Users\\Michał\\PycharmProjects\\PRO_1_Project\\Data\\grammatical_facial_expression')
    datapoint, target = split_datasets_names_in_directory(paths)
    points, cols = get_face_points(datapoint)
    print(points[0])
    print(cols)
