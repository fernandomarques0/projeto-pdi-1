import numpy as np
import cv2


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    """
    Garante que a imagem esteja em uint8 [0, 255]. Se não estiver, normaliza.
    """
    if img.dtype == np.uint8:
        return img
    img_f = img.astype(np.float32)
    mn = float(img_f.min())
    mx = float(img_f.max())
    img_f = img_f - mn
    if mx - mn > 1e-12:
        img_f = img_f * (255.0 / (mx - mn))
    return np.clip(img_f, 0, 255).astype(np.uint8)


def _equalize_channel(channel: np.ndarray) -> np.ndarray:
    """
    Equaliza um canal (8 bits) com equalizeHist.
    """
    return cv2.equalizeHist(channel)


def _local_stretch_channel(channel: np.ndarray, m: int, n: int) -> np.ndarray:
    """
    Expansão de histograma local: para cada pixel, aplica:
      out = (I - min_local) * 255 / (max_local - min_local)
    usando janela m x n. Quando max_local == min_local, mantém o valor original.
    """
    # Kernel retangular de tamanho (colunas=n, linhas=m)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n, m))

    # Mínimo e máximo locais via morfologia
    local_min = cv2.erode(channel, kernel)
    local_max = cv2.dilate(channel, kernel)

    ch_f = channel.astype(np.float32)
    min_f = local_min.astype(np.float32)
    max_f = local_max.astype(np.float32)

    denom = (max_f - min_f)
    # Evita divisão por zero
    safe_denom = denom.copy()
    safe_denom[safe_denom == 0] = 1.0

    out = (ch_f - min_f) * 255.0 / safe_denom
    # Onde não há variação local (max == min), mantém valor original
    mask_flat = (denom == 0)
    out[mask_flat] = ch_f[mask_flat]

    return np.clip(out, 0, 255).astype(np.uint8)


def equalize_and_local_stretch(img: np.ndarray, m: int, n: int) -> np.ndarray:
    """
    Aplica equalização global seguida de expansão (stretching) local do histograma
    com janela m x n nos canais R, G e B.

    Parâmetros:
        img: np.ndarray - imagem grayscale ou RGB(A). Valores serão convertidos para uint8 [0,255] se necessário.
        m: int - altura da janela (linhas).
        n: int - largura da janela (colunas).

    Retorna:
        np.ndarray - imagem resultante em uint8.
    """
    if m <= 0 or n <= 0:
        raise ValueError("m e n devem ser inteiros positivos.")

    # Trata grayscale
    if img.ndim == 2:
        ch = _ensure_uint8(img)
        ch_eq = _equalize_channel(ch)
        ch_out = _local_stretch_channel(ch_eq, m, n)
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
        r_out = _local_stretch_channel(r_eq, m, n)
        g_out = _local_stretch_channel(g_eq, m, n)
        b_out = _local_stretch_channel(b_eq, m, n)

        out_rgb = np.dstack([r_out, g_out, b_out]).astype(np.uint8)

        if has_alpha:
            alpha = _ensure_uint8(img[:, :, 3])
            out = np.dstack([out_rgb, alpha])
            return out.astype(np.uint8)

        return out_rgb.astype(np.uint8)

    raise ValueError("Formato de imagem não suportado. Use grayscale ou RGB(A).")