# TODO Deletar esse arquivo antes da apresentação
import matplotlib.pyplot as plt
import numpy as np


def load_image(path) -> np.ndarray:
    return plt.imread(path)


def save_image(img, path: str) -> None:
    plt.imsave(path, img)


def display_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def read_filter(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # First line: m n
    m, n = map(int, lines[0].split())

    # Next m lines: mask rows
    mask = []
    for i in range(1, 1 + m):
        row = list(map(float, lines[i].split()))
        if len(row) != n:
            raise ValueError("Dimensões do filtro incompatíveis.")
        mask.append(row)
    mask = np.array(mask)

    # Bias
    bias = int(lines[1 + m])
    if bias < -255 or bias > 255:
        raise ValueError("{} não é um valor aceitável como 'bias'.".format(bias))

    # Activation
    activation = lines[2 + m]
    if activation not in ['ReLU', 'Identity']:
        raise ValueError("A função de ativação deve ser 'ReLU' ou 'Identity'")

    return mask, bias, activation