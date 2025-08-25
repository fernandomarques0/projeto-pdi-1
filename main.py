import numpy as np
from PIL import Image

from fs import get_filter_from_file
from procedures import correlation
from procedures.histogram import equalize_and_local_stretch

FILE_PATH = "./test_images/frutas.png"
FILTERS_PATH = './filters/filtro-gauss.txt'
OUTPUT_PATH = "./results/"

def main():
    m, n, offset, i, _filter = get_filter_from_file(FILTERS_PATH)

    matriz = np.array(_filter)
    matriz = matriz.reshape(m, n)
    matriz = matriz.astype(float)

    correlation(FILE_PATH, OUTPUT_PATH + "correlation-gauss.png", m, n, matriz, i, offset)

    img = np.array(Image.open(OUTPUT_PATH + "correlation-gauss.png"))
    out = equalize_and_local_stretch(img, 7, 7)
    Image.fromarray(out).save("saida.png")

if __name__ == "__main__":
    main()