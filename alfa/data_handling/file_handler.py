import os


def get_datasets_names_in_directory(path: str) -> list[str]:
    """
    Metoda do pobrania ścieżek do plików z danymi: datapoints.txt i targets.txt
    z folderu ../grammatical_facial_expression i zwraca je w postaci listy
    :param path: Ścieżka do folderu ../grammatical_facial_expression
    :return: Lista wszystkich ścieżek do plików z danymi: datapoints.txt i targets.txt
    """
    if not os.path.isdir(path):
        raise NotADirectoryError(f'{path} is not a directory')

    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} does not exist')

    files: list[str] = [(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    datasets: list[str] = [f for f in files if f.endswith('targets.txt') or f.endswith('datapoints.txt')]

    if not datasets:
        raise FileNotFoundError(f'No dataset files in {path}')

    return datasets


def split_datasets_names_in_directory(paths: list[str]) -> dict[str, list[str]]:
    """

    :param paths: Lista ze ściażkami do plików z danymi: datapoints.txt i targets.txt
    :return: Słownik gdzie klucze to nazwa wyrazu(nazwa plik bez .txt), a wartościami są listy
    ze ścieżkami do odpowiednich plików datapoints.txt i targets.txt
    """
    datapoints: dict[str, list[str]] = {}

    for file_path in paths:
        current_face = file_path.split('\\')[-1].replace('.txt', '')
        if current_face.endswith('datapoints'):
            current_face = current_face.replace('_datapoints', '')
        elif current_face.endswith('targets'):
            current_face = current_face.replace('_targets', '')
        if current_face not in datapoints:
            datapoints[current_face] = [file_path]
        else:
            datapoints[current_face].append(file_path)

    return datapoints


def get_face_points(data_paths: dict[str, list[str]]) -> list[list]:
    """

    :param data_paths: słownik ze ścieżkami do plików z danymi: datapoints.txt i targets.txt
    :return: Lista list z danymi: punktami i targetem i nazwą wyrazu twarzy
    """
    data: list[list] = []

    for target, file_paths in data_paths.items():
        targets = file_paths[0] if file_paths[0].endswith('targets.txt') else file_paths[1]
        datapoints = file_paths[1] if file_paths[1].endswith('datapoints.txt') else file_paths[0]

        with open(targets, 'r') as t:
            with open(datapoints, 'r') as d:
                d.readline()
                for line in d.readlines():
                    l = list(map(float, line.strip().split(' ')))
                    l.append(int(t.readline().strip()))
                    l.append(target)
                    data.append(l)
            d.close()
        t.close()

    return data
