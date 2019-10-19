from skimage import io
from skimage import color
from skimage import exposure
from matplotlib import pyplot as plt


def hw4():
    image_path = './data/ax.jpg'

    img = io.imread(image_path)
    grey_img = color.rgb2gray(img) * 255

    histogram = exposure.histogram(grey_img)

    fig, axes = plt.subplots(1, 2, figsize=(30, 10))

    axes[0].imshow(grey_img, 'Greys_r')
    axes[0].set_title('Original Image')
    axes[1].hist(grey_img)
    axes[1].set_title('Histogram')

    fig.show()
