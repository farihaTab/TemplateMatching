import numpy as np
from scipy import ndimage, misc
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image
import matplotlib.pyplot as plt

# data = ndimage.imread("Data/test_image.png",True)
# # plt.imshow(data, cmap=plt.get_cmap('gray'))
# # plt.show()
# lowpass = ndimage.gaussian_filter(data, 1.5)
# # plt.imshow(lowpass, cmap=plt.get_cmap('gray'))
# # plt.show()
# image_rescaled = misc.imresize(data,.25)
# # plt.imshow(image_rescaled, cmap=plt.get_cmap('gray'))
# # plt.show()


img = ndimage.imread("Data/test_image.png")
plt.imshow(img, interpolation='nearest')
plt.show()
# Note the 0 sigma for the last axis, we don't wan't to blurr the color planes together!
img = ndimage.gaussian_filter(img, sigma=(1, 1, 0), order=0)
plt.imshow(img, interpolation='nearest')
plt.show()