import numpy as np
from PIL import Image

from fs import get_filter_from_file
from procedures import correlation
from procedures.histogram import equalize_and_local_stretch
from procedures.post_processing import post_process_sobel
from procedures.correlation import save_image

FILE_PATH = "./test_images/Shapes.png"
FILTERS_PATH = './filters/filtro-sobel-horizontal.txt'
OUTPUT_PATH = "./results/"
OUTPUT_PATH_SOBELS = "./results/sobels"

def main():
    m, n, offset, i, _filter = get_filter_from_file(FILTERS_PATH)

    matriz = np.array(_filter)
    matriz = matriz.reshape(m, n)
    matriz = matriz.astype(float)

    filtered_img = correlation(FILE_PATH, m, n, matriz, i, offset)

    # salva direto sem normalizar (pode parecer escuro)
    save_image(filtered_img, OUTPUT_PATH + "/correlation-sobel-horizontal.png")

    # aplica pós-processamento e salva normalizado
    sobel_processed = post_process_sobel(filtered_img)
    Image.fromarray(sobel_processed).save(OUTPUT_PATH + "/sobel-horizontal-processed.png")

    img = np.array(Image.open(FILE_PATH))
    out = equalize_and_local_stretch(img, 7, 7)
    Image.fromarray(out).save("saida.png")

if __name__ == "__main__":
    main()