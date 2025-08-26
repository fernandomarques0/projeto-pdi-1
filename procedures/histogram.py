import numpy as np
import cv2

def histogram_expansion(channel: np.ndarray) -> np.ndarray:
    """
    Expande o intervalo de intensidades de pixels em um canal de imagem para o
    padrão de 8 bits (0–255). Esta operação é comumente usada em processamento
    de imagens para aumentar o contraste, redistribuindo os valores de
    intensidade ao longo de toda a faixa.

    :param channel: Array 2D do NumPy contendo os valores de intensidade de
        pixels (por exemplo, um canal de imagem). Os valores podem ser inteiros
        ou floats.
    :return: Array 2D com o mesmo formato do canal de entrada, contendo os
        valores expandidos como inteiros sem sinal de 8 bits (np.uint8).
    """
    _min = channel.min()
    _max = channel.max()
    dif = _max - _min
    return np.zeros_like(channel, dtype=np.uint8) if dif == 0 else (
        np.clip(
            ((channel - _min) / dif if dif != 0 else 1) * 255,
            0, 255
        ).astype(np.uint8))


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    """
    Garante que o array de imagem de entrada tenha o tipo `np.uint8`. Se a imagem
    não for do tipo `uint8`, esta função normaliza seus valores para o intervalo
    [0, 255], escala adequadamente e converte o array para `np.uint8`.

    :param img: Imagem de entrada como um array NumPy. Pode ser de qualquer tipo
        numérico.
    :type img: np.ndarray
    :return: A imagem processada como um array NumPy com tipo `np.uint8`.
    :rtype: np.ndarray
    """
    if img.dtype == np.uint8:
        return img

    img_f = img.astype(np.float32)
    _min = float(img_f.min())
    _max = float(img_f.max())

    img_f = img_f - _min
    if _max - _min > 1e-12:
        img_f = img_f * (255.0 / (_max - _min))

    return np.clip(img_f, 0, 255).astype(np.uint8)


def _equalize_channel(channel: np.ndarray) -> np.ndarray:
    """
    Equaliza o histograma de um único canal, redistribuindo os níveis de intensidade
    ao longo do intervalo do canal. A função modifica o canal de entrada de forma que
    as intensidades de pixels sejam redistribuídas para melhorar o contraste, mantendo
    a estrutura geral da imagem.

    A função lida com casos em que o canal possui variação, é degenerado (vazio ou
    com um único nível de intensidade), ou quando o tipo de dados requer conversão
    para uint8.

    :param channel: Canal de imagem de entrada como um array 2D do NumPy. Deve ser do tipo
                    uint8 ou conversível para uint8.
    :return: Array 2D do mesmo formato do canal de entrada com o histograma equalizado.
             Tipo de dado é uint8.
    """
    if channel.dtype != np.uint8:
        channel = _ensure_uint8(channel)

    flat = channel.ravel()
    hist = np.bincount(flat, minlength=256).astype(np.int64)
    cdf = hist.cumsum()

    # Menor CDF > 0
    nonzero = cdf[cdf > 0]
    if nonzero.size == 0:
        # Imagem vazia (degenarada)
        return np.zeros_like(channel, dtype=np.uint8)

    cdf_min = int(nonzero[0])
    total = int(flat.size)
    denom = total - cdf_min

    # Se não há variação (uma única intensidade), preserva a imagem
    if denom <= 0:
        return channel.copy()

    # LUT com arredondamento ao inteiro mais próximo (similar ao cvRound)
    lut = np.rint((cdf - cdf_min) * 255.0 / denom).astype(np.int32)
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    return lut[channel]


def _local_histogram_expansion(channel: np.ndarray, m: int, n: int) -> np.ndarray:
    """
    Aplica alongamento (stretching) de contraste local ao canal de entrada usando
    operações morfológicas. A intensidade é ajustada com base nos valores mínimo
    e máximo locais em uma região definida pelo kernel.

    O algoritmo obtém o mínimo e o máximo locais via erosão e dilatação. Nas áreas
    onde não há variação (isto é, mínimo igual ao máximo), os valores originais são
    preservados, garantindo que o ajuste de contraste ocorra apenas onde há
    variação.

    :param channel: Canal de imagem de entrada representado como um array 2D do NumPy.
    :param m: Altura do kernel retangular usado nos cálculos locais.
    :param n: Largura do kernel retangular usado nos cálculos locais.
    :return: Array 2D do mesmo formato da entrada, com valores ajustados de contraste.
    """
    # Kernel retangular de tamanho (colunas=n, linhas=m)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n, m))

    # Mínimo e máximo locais via morfologia
    local_min = cv2.erode(channel, kernel)
    local_max = cv2.dilate(channel, kernel)

    ch_f = channel.astype(np.float32)
    min_f = local_min.astype(np.float32)
    max_f = local_max.astype(np.float32)

    dif = (max_f - min_f)
    # Evita divisão por zero
    safe_denom = dif.copy()
    safe_denom[safe_denom == 0] = 1.0

    out = (ch_f - min_f) * 255.0 / safe_denom
    # Onde não há variação local (max == min), mantém valor original
    mask_flat = (dif == 0)
    out[mask_flat] = ch_f[mask_flat]

    return np.clip(out, 0, 255).astype(np.uint8)


def equalize_and_local_expansion(img: np.ndarray, m: int, n: int) -> np.ndarray:
    """
    Aplica equalização global de histograma e alongamento local de intensidade em
    uma imagem.

    A função ajusta a distribuição de intensidades pela equalização de histograma
    global seguida do alongamento local de contraste. Suporta imagens em tons de
    cinza e imagens RGB(A). Imagens em tons de cinza são processadas como um
    todo; em imagens RGB(A), cada canal de cor é processado independentemente. Se
    houver canal alfa, ele é preservado, mas não é processado.

    :param img: Imagem de entrada a ser processada. Pode ser grayscale ou RGB(A).
    :type img: numpy.ndarray
    :param m: Altura da região local para o alongamento de intensidade. Deve ser
        um inteiro positivo.
    :type m: int
    :param n: Largura da região local para o alongamento de intensidade. Deve ser
        um inteiro positivo.
    :type n: int
    :return: Imagem processada com equalização de intensidades e alongamento
        local aplicados, mantendo as mesmas dimensões da imagem de entrada.
    :rtype: numpy.ndarray
    :raises ValueError: Se `m` ou `n` não for inteiro positivo, ou se o formato
        da imagem de entrada não for suportado.
    """
    if m <= 0 or n <= 0:
        raise ValueError("m e n devem ser inteiros positivos.")

    # Trata grayscale
    if img.ndim == 2:
        ch = _ensure_uint8(img)
        ch_eq = _equalize_channel(ch)
        ch_out = _local_histogram_expansion(ch_eq, m, n)
        return ch_out

    # Trata RGB(A)
    if img.ndim == 3:
        # Se houver canal alfa, preserva
        has_alpha = img.shape[2] == 4
        rgb = img[..., :3]
        rgb = _ensure_uint8(rgb)

        # Separa R, G, B preservando a ordem natural do array
        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]

        # Equalização global por canal
        r_eq = _equalize_channel(r)
        g_eq = _equalize_channel(g)
        b_eq = _equalize_channel(b)

        # Expansão local por canal
        r_out = _local_histogram_expansion(r_eq, m, n)
        g_out = _local_histogram_expansion(g_eq, m, n)
        b_out = _local_histogram_expansion(b_eq, m, n)

        out_rgb = np.dstack([r_out, g_out, b_out]).astype(np.uint8)

        if has_alpha:
            alpha = _ensure_uint8(img[:, :, 3])
            out = np.dstack([out_rgb, alpha])
            return out.astype(np.uint8)

        return out_rgb.astype(np.uint8)

    raise ValueError("Formato de imagem não suportado. Use grayscale ou RGB(A).")