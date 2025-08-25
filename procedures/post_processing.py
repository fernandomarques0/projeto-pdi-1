import numpy as np


def post_process_sobel(filtered_img):
    abs_img = np.abs(filtered_img)
    result = np.zeros_like(abs_img)
    for c in range(3):
        channel = abs_img[:, :, c]
        min_val = channel.min()
        max_val = channel.max()
        if max_val - min_val == 0:
            result[:, :, c] = 0
        else:
            result[:, :, c] = (channel - min_val) / (max_val - min_val) * 255
    return result.astype(np.uint8)