# method-1 to convert to grayscale
from scipy import ndimage
data = ndimage.imread("Data/ref_image.jpg",True)
print(data.shape)
print(data)

# method-2 to convert to grayscale using mathplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = mpimg.imread('G:\Projects\PycharmProjects\PatternRecognition\TemplateMatching\Data\\ref_image.jpg')
gray = rgb2gray(img)
print(gray.shape)
print(gray)
# plt.imshow(gray, cmap = plt.get_cmap('gray'))
# plt.show()

#method-3 to convert to grayscale
from PIL import Image
img = Image.open('Data/ref_image.jpg').convert('LA')
img.save('greyscale.png')

x1 = np.arange(9.0).reshape((3, 3))
print(x1)
x2 = np.arange(9.0).reshape((3, 3))
print(x2)
print(np.multiply(x1, x2))
print(x1*x2)
print(np.dot(x1,x2))