from scipy import ndimage, misc
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import math

arr = np.arange(5*3*4).reshape((5,3,4))
print(arr)

print(arr.shape)

print(arr[1:3,0:3,:])
print(arr[1:3,0:3,:]**2)
print(arr[1:3,0:3,:]*arr[1:3,0:3,:])

from scipy import ndimage, misc
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys, os
import math
im = Image.open("Data/test_image.png")
print(np.array(im).shape)
rgb = Image.new("RGB",im.size,(255,255,255))
rgb.paste(im, (0,0), im)
rgb = np.array(rgb)
print(rgb.shape)
plt.imshow(rgb)
# plt.show()
im = Image.open("Data/test_2.jpg")
im = ndimage.uniform_filter(im)
im= misc.imresize(im,0.2)
plt.imshow(im)
plt.show()