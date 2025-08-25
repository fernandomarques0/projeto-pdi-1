from typing import Any

import numpy as np
from PIL import Image
from PIL.ImageFile import ImageFile


def open_image(path: str) -> ImageFile:
    image = Image.open(path)
    return image


def get_filter_from_file(filepath: str) -> tuple[int, int, int, str, np.ndarray]:
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # First line: m n
    rows, columns = map(int, lines[0].split())

    # Next m lines: mask rows
    mask = []
    for i in range(1, 1 + rows):
        row = list(map(float, lines[i].split()))
        if len(row) != columns:
            raise ValueError("Dimensões do filtro incompatíveis.")
        mask.append(row)
    _filter = np.array(mask)
    print(_filter)

    # Bias
    bias = int(lines[1 + rows])
    if bias < -255 or bias > 255:
        raise ValueError("{} não é um valor aceitável como 'bias'.".format(bias))

    # Activation
    activation = lines[2 + rows]
    if activation not in ['ReLU', 'Identity']:
        raise ValueError("A função de ativação deve ser 'ReLU' ou 'Identity'")

    return rows, columns, bias, activation, _filter
