from typing import Any

import numpy as np
from PIL import Image
from PIL.ImageFile import ImageFile


def open_image(path: str) -> ImageFile:
    """
    Abre um arquivo de imagem a partir de um caminho especificado.

    Esta função utiliza a biblioteca PIL para abrir uma imagem localizada no
    caminho informado e retorna o objeto de imagem aberto.

    :param path: Caminho do arquivo da imagem a ser aberta.
    :type path: str
    :return: Objeto que representa a imagem aberta.
    :rtype: ImageFile
    """
    image = Image.open(path)
    return image


def save_image(array: np.ndarray, path: str):
    """
    Esta função recebe um array NumPy representando dados de imagem, limita os
    valores de pixels ao intervalo [0, 255], converte para inteiro sem sinal de
    8 bits e salva a imagem no caminho especificado.

    :param array: Array NumPy contendo os dados da imagem a ser salva.
    :param path: Caminho do arquivo onde a imagem será salva.
    :return: None
    """
    img_uint8 = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)


def get_filter_from_file(filepath: str) -> tuple[int, int, int, str, np.ndarray]:
    """
    Lê de um arquivo os parâmetros e a máscara de um filtro. O arquivo deve conter
    as dimensões do filtro, as linhas da máscara, o bias e a função de ativação.
    São feitas validações para garantir dimensões e parâmetros válidos.

    :param filepath: Caminho do arquivo contendo os dados do filtro.
    :type filepath: str
    :return: Tupla com (linhas, colunas, bias, função de ativação, máscara como np.ndarray).
    :rtype: tuple[int, int, int, str, np.ndarray]
    :raises ValueError: Se as dimensões das linhas da máscara não corresponderem ao número de colunas especificado.
    :raises ValueError: Se o valor de bias estiver fora do intervalo aceitável (-255 <= bias <= 255).
    :raises ValueError: Se a função de ativação não for 'ReLU' ou 'Identity'.
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    print("Filter read from: {}".format(filepath))

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

    # Bias
    bias = int(lines[1 + rows])
    if bias < -255 or bias > 255:
        raise ValueError("{} não é um valor aceitável como 'bias'.".format(bias))

    # Activation
    activation = lines[2 + rows]
    if activation not in ['ReLU', 'Identity']:
        raise ValueError("A função de ativação deve ser 'ReLU' ou 'Identity'")

    print("filter {}x{}".format(rows, columns))
    print("With bias = {} and activation function = {}".format(bias, activation))
    print(_filter)

    return rows, columns, bias, activation, _filter
