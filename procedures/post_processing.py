import numpy as np
from procedures import histogram_expansion

def post_process_sobel(filtered_img):
    """
    Pós-processa uma imagem filtrada por Sobel aplicando expansão de histograma em
    cada canal de cor e convertendo a imagem resultante para o formato inteiro
    sem sinal de 8 bits.

    :param filtered_img: Imagem de entrada filtrada por Sobel como um array NumPy
        3D no formato (altura, largura, canais). Os valores devem representar
        gradientes filtrados.
    :return: Array NumPy 3D com o mesmo formato da entrada, com canais de cor
        expandidos por histograma e convertidos para uint8.
    """
    abs_img = np.abs(filtered_img)

    channel_r = abs_img[:, :, 0]
    channel_g = abs_img[:, :, 1]
    channel_b = abs_img[:, :, 2]

    return np.dstack([
        histogram_expansion(channel_r),
        histogram_expansion(channel_g),
        histogram_expansion(channel_b)
    ]).astype(np.uint8)