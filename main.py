import numpy as np
from PIL import Image

from fs import get_filter_from_file, save_image
from procedures import correlation, post_process_sobel, equalize_and_local_expansion

FILE_PATH = "./test_images/Shapes.png"
FILTERS_PATH = './filters/filtro-sobel-vertical.txt'
OUTPUT_PATH = "./results/"

def main():
    m, n, bias, activation, _filter = get_filter_from_file(FILTERS_PATH)
    export_file_name = "{}-{}-{}".format(FILE_PATH.split("/")[-1].split(".")[0], FILTERS_PATH.split("/")[-1].split(".")[0], activation)
    export_file_to_equalize = "{}".format(FILE_PATH.split("/")[-1].split(".")[0])

    filtered_img = correlation(FILE_PATH, m, n, _filter, activation, bias)

    # salva direto sem normalizar
    save_image(filtered_img, OUTPUT_PATH + "/correlation/" + export_file_name + ".png")

    # aplica pós-processamento e salva normalizado
    sobel_processed = post_process_sobel(filtered_img)
    Image.fromarray(sobel_processed).save(OUTPUT_PATH + "/sobel/"+ export_file_name + "-processed.png")

    img = np.array(Image.open(FILE_PATH))
    out = equalize_and_local_expansion(img, 100, 100)
    Image.fromarray(out).save(OUTPUT_PATH + "/equalize/" + export_file_to_equalize + "-equalize&local_expansion.png")

if __name__ == "__main__":
    main()