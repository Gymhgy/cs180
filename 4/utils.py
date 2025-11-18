import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d
import numpy as np
import skimage.io as skio

def gaussian_filter(sigma=0, ksize=5):
    g1d = cv2.getGaussianKernel(ksize, sigma)
    return g1d @ g1d.T

def apply_to_channels(im, func):
    im_out = np.zeros_like(im)
    for c in range(im.shape[2]):
        im_out[:, :, c] = func(im[:, :, c])
    return im_out

def show_image(im, filename=None):
    # if the img is in [0, 1], scale to [0, 255]
    if im.dtype == np.float32 or im.dtype == np.float64:
        im = (im * 255).astype(np.uint8)
    if filename:
        skio.imsave(filename, im)
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.show()