import sys

import numpy as np
import cv2
from PIL import Image


def expansion_by_zero(matriz, rows: int, columns: int) -> np.ndarray:
    """
    Aplica a expansão por zero a uma matriz caso necessário.

    :param matriz: A matriz de entrada que será expandida.
    :type matriz: numpy.ndarray
    :param rows: Número de linhas desejado na matriz expandida.
    :param columns: Número de colunas desejado na matriz expandida.
    :return: A matriz expandida com as dimensões especificadas, preenchida com zeros
             nas posições extra.
    :rtype: numpy.ndarray
    """
    _rows, _columns = matriz.shape
    matriz_res = np.zeros((rows, columns))
    matriz_res[:_rows, :_columns] = matriz

    return matriz_res


def _apply_activation(value: float, activation: str) -> float:
    """
    Aplica a função de ativação especificada a um valor numérico.

    :param value: Valor numérico de entrada ao qual a função de ativação será aplicada.
    :type value: float
    :param activation: Tipo de função de ativação como string. Valores suportados incluem 'ReLU'.
    :type activation: str
    :return: Valor numérico após a aplicação da função de ativação.
    :rtype: float
    """
    return max(0.0, float(value)) if activation == 'ReLU' else float(value)


def _compute_pixel(channel: np.ndarray, x: int, y: int, rows: int, columns: int, _filter: np.ndarray, activation: str, bias: int) -> float:
    """
    Computa o valor de um único pixel em uma imagem processada aplicando um filtro,
    bias e função de ativação a um recorte (patch) da imagem de entrada.

    :param channel: Array 2D do NumPy representando o canal da imagem onde a
        operação será aplicada.
    :param x: Coordenada x (índice de linha) do canto superior esquerdo do patch a
        ser processado.
    :param y: Coordenada y (índice de coluna) do canto superior esquerdo do patch a
        ser processado.
    :param rows: Número de linhas do patch a extrair e processar.
    :param columns: Número de colunas do patch a extrair e processar.
    :param _filter: Array 2D do NumPy representando o filtro a ser aplicado no patch.
    :param activation: Nome da função de ativação a aplicar ao valor computado.
    :param bias: Valor de bias somado ao resultado antes da ativação.

    :return: Valor float resultante da aplicação do filtro, bias e função de
        ativação sobre o patch de entrada.
    """
    patch = channel[x:x + rows, y:y + columns]
    matriz = expansion_by_zero(patch, rows, columns)
    value = float(np.sum(matriz * _filter) + bias)
    return _apply_activation(value, activation)


def correlation(image_path: str, rows: int, columns: int, _filter: np.ndarray, activation: str, bias: int):
    """
    Executa a operação de correlação 2D em uma imagem usando o filtro, a função
    de ativação e o bias especificados. A função processa a imagem por canal
    (Vermelho, Verde, Azul) e aplica a correlação pixel a pixel.

    :param image_path: Caminho para o arquivo de imagem de entrada.
    :type image_path: str
    :param rows: Número de linhas do filtro de convolução/correlação.
    :type rows: int
    :param columns: Número de colunas do filtro de convolução/correlação.
    :type columns: int
    :param _filter: Array 2D do NumPy representando o filtro a ser aplicado.
    :type _filter: np.ndarray
    :param activation: Função de ativação a aplicar ao resultado da correlação.
    :type activation: str
    :param bias: Valor de bias a adicionar ao resultado antes da ativação.
    :type bias: int
    :return: Array de imagem em ponto flutuante contendo os resultados por canal.
    :rtype: np.ndarray
    """
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
