import torch


def standarise_data(data: torch.Tensor) -> torch.tensor:
    """
    Metoda do standaryzacji danych na podstawwie wspołrzędnych X, Y i Z
    :param data: Zbiór danych
    :return: Standaryzowany zbiór danych
    """
    x_data = data[:, 0::3]
    y_data = data[:, 1::3]
    z_data = data[:, 2::3]

    x_mean = x_data.mean()
    y_mean = y_data.mean()
    z_mean = z_data.mean()

    x_std = x_data.std()
    y_std = y_data.std()
    z_std = z_data.std()

    normalised_data = data.clone()

    normalised_data[:, 0::3] = (x_data - x_mean) / x_std
    normalised_data[:, 1::3] = (y_data - y_mean) / y_std
    normalised_data[:, 2::3] = (z_data - z_mean) / z_std

    return normalised_data


def filter_dataset(data: list[list[float]]) -> list[list[float]]:
    """
    Metoda do odsiania danych na podstawie tego czy jest wykrywany gest
    :param data: Zbiór danych
    :return: Zbiór danych bez wykrytych gestów
    """
    filtered_data: list[list[float]] = []
    for row in data:
        if row[-2] == 1:
            filtered_data.append(row)
    return filtered_data

def split_data(data: list[list[float]], categories: dict) -> tuple:
    """
    Metoda do podzielenia zbioru danych na dane i target
    :param categories: Mapa kategorii
    :param data: Zbiór danych
    :return: Dane i target
    """
    points: list[list[float]] = []
    targets: list[int] = []

    for row in data:
        points.append(row[1:-2])
        targets.append(categories[row[-1]])

    return points, targets

def to_pytorch_tensor(data: list[list[float]], target: list[int]) -> tuple:
    """
    Metoda do konwersji danych do formatu pytorch tensor
    :param target: Target
    :param data: Zbiór danych
    :return: Dane i target w postaci pytorch tensor
    """
    tensor_data = torch.Tensor(data).float()
    tensor_target = torch.Tensor(target).long()
    return tensor_data, tensor_target