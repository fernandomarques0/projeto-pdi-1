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
