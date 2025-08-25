import numpy as np

from procedures.correlation import correlate_v2


def _define_equacao_da_reta() -> tuple[float, float]:
    #Calcula inclinação m
    m = (0-255)/(255-128)

    #Calcula coeficiente linear b
    b = 255 - (m*128)

    return m, b


def _pixel_maior_que_128(pixel, m, b) -> float:
    y = m*pixel + b
    return y

def _pixel_menor_igual_128(pixel) -> float:
    y = (pixel*255)/128
    return y


def filtro_pontual() -> float:
    pixel = 150  # Valor do pixel de exemplo
    m, b = _define_equacao_da_reta()  # Armazena os valores da reta de descida do filtro pontual

    if pixel <= 128:
        valor = _pixel_menor_igual_128(pixel)  # Faz uma regra de três simples para encontrar o valor na primeira reta.
    else:
        valor = _pixel_maior_que_128(pixel, m, b)  # Faz o calculo baseado na equação da reta.

    return valor


# Não funciona
def apply_filter(img, mask, bias, activation):
    r = correlate_v2(img[:, :, 0], mask, bias, activation)
    g = correlate_v2(img[:, :, 1], mask, bias, activation)
    b = correlate_v2(img[:, :, 2], mask, bias, activation)
    return np.stack([r, g, b], axis=2)