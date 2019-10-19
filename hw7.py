from skimage.io import imread
from matplotlib import pyplot as plt
import numpy as np

RANGE = 256


# utility function plotting histogram
def plot_hist(img, title, pixel_value_range = RANGE):
    image_hist, bins = np.histogram(img, bins=range(pixel_value_range))
    center = (bins[:-1] + bins[1:]) / 2
    plt.title(title)
    plt.bar(center, image_hist, width=1)
    plt.show()


# utility function plotting image
def grey_image_show(img, title):
    plt.title(title)
    plt.imshow(img, 'Greys_r')
    plt.show()


# get a cdv (discrete cdf) of an image
def get_cumulative_density_vector(img, value_range=RANGE):
    hist, bins = np.histogram(img, bins=value_range, range=(0, value_range))
    h_r = hist / img.size
    return np.cumsum(h_r)


# get cdv of histogram equalization
def get_histogram_equalization_density_vector(value_range=RANGE):
    h_r = np.ones((value_range,), dtype=np.float64)
    h_r = np.divide(h_r, value_range)
    return np.cumsum(h_r)


# make a lookup table for transforming a source image to desired image base on their CDVs'
# CDV: discrete CDF or cumulative density function.
def get_image_transform_lookup_table(source_image_cdv, target_image_cdv, value_range=RANGE):
    lookup = np.ndarray((value_range,))
    for i in range(value_range):
        diff = 10
        prominent_j = 0
        for j in range(value_range):
            j_diff = np.abs(source_image_cdv[i] - target_image_cdv[j])
            if j_diff <= diff:
                diff = j_diff
                prominent_j = j
            else:
                break
        lookup[i] = prominent_j
    return lookup


# image transformer based on target CDV
def transform_image_via_cdv(img, target_image_cdv, value_range=RANGE):
    source_cdv = get_cumulative_density_vector(img, value_range)
    look_up_table = get_image_transform_lookup_table(source_cdv, target_image_cdv, value_range)

    def t(pixel):
        return look_up_table[int(pixel)]

    vt = np.vectorize(t)

    transformed_image = vt(img)
    return transformed_image


# special case of @transform_image_via_cdv where target_image_cdv is equalized CDV
def histogram_equalization(img, value_range=RANGE):
    H_s = get_histogram_equalization_density_vector(value_range)

    transformed_image = transform_image_via_cdv(img, H_s, value_range)

    return transformed_image


def main():
    image = np.floor(imread('./data/ax.jpg', as_gray=True) * RANGE) - 1  # read image as grey and make the values

    # segmenting image horizontally
    image_seg = np.array_split(image, 2)

    # each row
    for i in range(2):
        # segment each row vertically
        image_seg[i] = np.array_split(image_seg[i], 3, 1)
        # each segment
        for j in range(3):
            # histogram equalization
            image_seg[i][j] = histogram_equalization(image_seg[i][j], RANGE)
        # joining segment of each row
        image_seg[i] = np.concatenate(image_seg[i], 1)

    # join each row to make the image
    locally_hist_eq_transformed_image = np.concatenate(image_seg)
    # globally hist eq
    globally_hist_eq_transformed_image = histogram_equalization(image, RANGE)

    # showing the results
    grey_image_show(image, 'Orig')
    grey_image_show(locally_hist_eq_transformed_image, 'Locally')
    grey_image_show(globally_hist_eq_transformed_image, 'Globally')
    plot_hist(image, 'Orig')
    plot_hist(locally_hist_eq_transformed_image, 'Locally')
    plot_hist(globally_hist_eq_transformed_image, 'Globally')




