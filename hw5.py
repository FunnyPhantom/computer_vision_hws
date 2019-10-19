from skimage.io import imread
from matplotlib import pyplot as plt
import numpy as np


def hw5():
    img_path = './data/ax.jpg'
    img_grey = np.round(imread(img_path, True) * 255)

    rows, cols = img_grey.shape

    shp = [rows * cols, 3]
    vector_img = np.ndarray(shape=shp)

    i = 0
    for row in img_grey:
        vec_row = np.ndarray(shape=[cols, 3])
        vec_row[:, 0] = row
        vec_row[:, 1] = i
        vec_row[:, 2] = range(row.shape[0])
        vector_img[i * cols: (i + 1) * cols] = vec_row
        i = i + 1

    vec_sort = vector_img[vector_img[:, 0].argsort()]

    flat_hist_bar_len = int((vec_sort.shape[0] / 256))

    for i in range(256):
        vec_sort[i * flat_hist_bar_len: (i + 1) * flat_hist_bar_len, 0] = i
    flat_img = np.ndarray(shape=img_grey.shape)

    for cell in vec_sort:
        flat_img[int(cell[1]), int(cell[2])] = cell[0]

    hist = np.ndarray(shape=256, dtype=np.uint32)
    ii = 0
    for row in flat_img:
        for cell in row:
            if (cell == 12):
                ii = ii + 1

            hist[int(cell)] = hist[int(cell)] + 1

    fig, axes = plt.subplots(2, 2, figsize=(40, 40))
    axes[0][0].imshow(img_grey, 'Greys_r')
    axes[0][0].set_title('original image')
    axes[0][1].hist(img_grey)
    axes[0][1].set_title('original histogram')
    axes[1][0].imshow(flat_img, 'Greys_r')
    axes[1][1].hist(flat_img)
    fig.show()
