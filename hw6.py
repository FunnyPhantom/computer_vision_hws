from skimage.io import imread
from matplotlib import pyplot as plt
import numpy
from scipy import stats

RANGE = 256


# utility function plotting histogram
def plot_hist(img, title, pixel_value_range = RANGE):
    image_hist, bins = numpy.histogram(img, bins=range(pixel_value_range))
    center = (bins[:-1] + bins[1:]) / 2
    plt.title(title)
    plt.bar(center, image_hist, width=1)
    plt.show()


# utility function plotting image
def grey_image_show(img, title):
    plt.title(title)
    plt.imshow(img, 'Greys_r')
    plt.show()


# g(x,y) = T[f(x,y)] = L - f(x,y)
def hist_flip_transform_function(pixel, pixel_value_range):
    return pixel_value_range - pixel


# flip whole image
def hist_flip(img, pixel_value_range=RANGE):
    return hist_flip_transform_function(img, pixel_value_range)


# get CDF of an image as a discrete function
def get_cumulative_density_vector(img, value_range=RANGE):
    hist, bins = numpy.histogram(img, bins=value_range, range=(0, value_range))
    h_r = hist / img.size
    return numpy.cumsum(h_r)


# get exponential CDF over range as a discrete function
def get_exponential_cumulative_density_vector(exp_base, value_range=RANGE):
    nu = numpy.float64(exp_base ** value_range) / (exp_base - 1)
    alpha = numpy.divide(1, nu - 1)

    def ps(j):
        return numpy.multiply(alpha, numpy.float64(exp_base ** j))

    h_s = ps(numpy.linspace(0, RANGE - 1, RANGE))
    return numpy.cumsum(h_s)


# main functionality of hw6_a
def hw6_a(img):
    transformed_image = hist_flip(img)

    grey_image_show(img, 'Original')
    plot_hist(img, 'Original Histogram')
    grey_image_show(transformed_image, 'Transformed image')
    plot_hist(transformed_image, 'Transformed histogram')


# main functionality of hw6_b
def hw6_b(img, exp_base):

    H_r = get_cumulative_density_vector(img)
    H_s = get_exponential_cumulative_density_vector(exp_base)

    lookup = numpy.ndarray((RANGE,))

    for i in range(RANGE):
        diff = 10
        prominent_j = 0
        for j in range(RANGE):
            j_diff = numpy.abs(H_r[i] - H_s[j])
            if j_diff <= diff:
                diff = j_diff
                prominent_j = j
            else:
                break
        lookup[i] = prominent_j

    def t(pixel):
        return lookup[int(pixel)]

    vt = numpy.vectorize(t)

    transformed_image = vt(img)

    grey_image_show(transformed_image, 't_img')
    plot_hist(transformed_image, 'hist')


def main():
    image = numpy.floor(imread('./data/ax.jpg', as_gray=True) * RANGE) - 1
    hw6_a(image)
    hw6_b(image, exp_base=1.05)
