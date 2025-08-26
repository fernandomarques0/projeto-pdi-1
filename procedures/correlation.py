import sys

import numpy as np
import cv2
from PIL import Image


def _transform(matriz, rows: int, columns: int) -> np.ndarray:
    _rows, _columns = matriz.shape
    matriz_res = np.zeros((rows, columns))
    matriz_res[:_rows, :_columns] = matriz

    return matriz_res


def _apply_activation(value: float, activation: str) -> float:
    if activation == 'ReLU':
        return max(0.0, float(value))
    elif activation == 'Identity' or activation is None:
        return float(value)
    else:
        raise ValueError(f"Função de ativação inválida: {activation}")

def _compute_pixel(channel: np.ndarray, x: int, y: int, rows: int, columns: int, _filter: np.ndarray, activation: str, bias: int) -> float:
    patch = channel[x:x + rows, y:y + columns]
    transformed = _transform(patch, rows, columns)
    value = float(np.sum(transformed * _filter) + bias)
    value = _apply_activation(value, activation)
    return value


def correlation(image_path: str, rows: int, columns: int, _filter: np.ndarray, activation: str, bias: int):
    imagem = cv2.imread(image_path)

    if imagem is None:
        print("Erro na leitura da imagem")
        sys.exit(1)

    height, width, canais = imagem.shape
    img_float = np.zeros_like(imagem, dtype=float)

    # Divide canais
    b_channel, g_channel, r_channel = cv2.split(imagem)

    for x in range(height):
        for y in range(width):
            pixel_r = _compute_pixel(r_channel, x, y, rows, columns, _filter, activation, bias)
            pixel_g = _compute_pixel(g_channel, x, y, rows, columns, _filter, activation, bias)
            pixel_b = _compute_pixel(b_channel, x, y, rows, columns, _filter, activation, bias)

            img_float[x, y] = (pixel_r, pixel_g, pixel_b)

    return img_float  # retorna a matriz crua (float, ainda não salva)


def save_image(array: np.ndarray, path: str):
    """
    Salva a matriz como imagem uint8.
    """
    img_uint8 = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)


# Não funciona
def correlate_v2(channel, mask, bias, activation):
    h, w = channel.shape
    mh, mw = mask.shape
    pad_h = mh // 2
    pad_w = mw // 2

    # Zero padding
    padded = np.pad(channel.astype(float), ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros((h, w), dtype=float)

    for i in range(h):
        for j in range(w):
            patch = padded[i:i + mh, j:j + mw]
            val = np.sum(patch * mask) + bias
            if activation == 'ReLU':
                val = max(0, val)
            # Identity does nothing
            output[i, j] = val

    return output
