from scipy.signal import convolve2d
import numpy as np
from imageio import imread
import skimage.color
from scipy import ndimage

GRAYSCALE_MAX = 255
GRAYSCALE = 1
RGB = 2
RGB_DIM = 3
MIN_DM = 16


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def normalize(image):
    """
    This function changing the matrix values to float and between [0,1]
    """
    image = image.astype(np.float64)
    image /= GRAYSCALE_MAX
    return image


## PREV ##
def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    image = imread(filename)  # Image as matrix
    image = normalize(image)  # Matrix as float and normalized
    if len(image.shape) == RGB_DIM and representation == GRAYSCALE:  # in case of RGB2GRAY
        return skimage.color.rgb2gray(image)
    return image


def build_filter(filter_size):
    """
     build the normalized filter_vec according to filter_size
    :param filter_size:
    :return: filter_vec
    """
    base_vec = np.array([1, 1])
    filter_vec = base_vec
    for i in range(filter_size - 2):
        filter_vec = np.convolve(filter_vec, base_vec)
    filter_vec = filter_vec / filter_vec.sum()
    return filter_vec.reshape((1, filter_size))


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    im = ndimage.convolve(im, blur_filter)  # Row conv
    im = ndimage.convolve(im, blur_filter.T)  # Cols conv
    im = im[::2]  # Reduce rows
    im = im.transpose()
    im = im[::2]
    return im.transpose()

def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    filter_vec = build_filter(filter_size)
    pyr = []
    pyr.append(im)
    for i in range(max_levels - 1):
        current_im = reduce(pyr[-1], filter_vec)
        if current_im.shape[0] < MIN_DM or current_im.shape[1] < MIN_DM:
            break
        pyr.append(current_im)
    return pyr, filter_vec
