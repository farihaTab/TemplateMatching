import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage, misc

MAX_COST = 34242421311212313


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def getImageArray(img_path):
    if ".jpg" in img_path:
        return np.array(Image.open(img_path), dtype=int)
    elif ".png" in img_path:
        im = Image.open(img_path)
        image = Image.new("RGB", im.size, (255, 255, 255))
        image.paste(im, (0, 0), im)
        return np.array(image, dtype=int)


class TM:
    def __init__(self, ref_img, test_img):
        if type(ref_img) is str and type(test_img) is str:
            self.ref_img_array_gray = ndimage.imread(ref_img, True)
            self.test_img_array_gray = ndimage.imread(test_img, True)
        else:
            self.ref_img_array_gray = ref_img
            self.test_img_array_gray = test_img
        print(self.ref_img_array_gray.shape)
        print(self.test_img_array_gray.shape)
        if self.ref_img_array_gray.shape[0] > self.test_img_array_gray.shape[0] \
                or self.ref_img_array_gray.shape[1] > self.test_img_array_gray.shape[1]:
            raise Exception("shape error")
        self.M = self.ref_img_array_gray.shape[0]
        self.N = self.ref_img_array_gray.shape[1]
        self.I = self.test_img_array_gray.shape[0]
        self.J = self.test_img_array_gray.shape[1]

    def exhaustive_search(self):

        min_m = 0
        min_n = 0
        min_cost = MAX_COST
        for m in range(self.I - self.M + 1):
            for n in range(self.J - self.N + 1):
                cost = self.calculateMismatch(m, n)
                print(m, n, cost)
                if min_cost > cost:
                    min_cost = cost
                    min_m = m
                    min_n = n

        print('exhaustive res', min_m, min_n)
        plt.imshow(self.test_img_array_gray, cmap=plt.get_cmap('gray'))
        plt.show()
        gray = self.test_img_array_gray[min_m:min_m + self.M, min_n:min_n + self.N]
        plt.imshow(gray, cmap=plt.get_cmap('gray'))
        plt.show()
        return min_m, min_n

    def two_dim_logarithmic_search(self):
        min_m = 0
        min_n = 0
        min_cost = MAX_COST
        center_m = int(self.I / 2)
        center_n = int(self.J / 2)
        search_region_width_m = int(center_m / 2)
        search_region_width_n = int(center_n / 2)
        print('center', center_m, center_n)
        print('width', search_region_width_m, search_region_width_n)

        while search_region_width_m > 1 and search_region_width_n > 1:
            m = center_m
            n = center_n
            min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n, search_region_width_m,
                                                           search_region_width_n)
            m = center_m - search_region_width_m
            n = center_n - search_region_width_n
            min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n, search_region_width_m,
                                                           search_region_width_n)
            m = center_m
            n = center_n - search_region_width_n
            min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n, search_region_width_m,
                                                           search_region_width_n)
            m = center_m + search_region_width_m
            n = center_n - search_region_width_n
            min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n, search_region_width_m,
                                                           search_region_width_n)
            m = center_m - search_region_width_m
            n = center_n
            min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n, search_region_width_m,
                                                           search_region_width_n)
            m = center_m + search_region_width_m
            n = center_n
            min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n, search_region_width_m,
                                                           search_region_width_n)
            m = center_m - search_region_width_m
            n = center_n + search_region_width_n
            min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n, search_region_width_m,
                                                           search_region_width_n)
            m = center_m
            n = center_n + search_region_width_n
            min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n, search_region_width_m,
                                                           search_region_width_n)
            m = center_m + search_region_width_m
            n = center_n + search_region_width_n
            min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n, search_region_width_m,
                                                           search_region_width_n)
            if center_m == min_m and center_n == min_n:
                search_region_width_m = int(search_region_width_m / 2)
                search_region_width_n = int(search_region_width_n / 2)
            center_m = min_m
            center_n = min_n
            print('center', center_m, center_n)
            print('width', search_region_width_m, search_region_width_n)
            print()
        print('res', min_m, min_n)
        # gray = self.test_img_array_gray[min_m:min_m + self.M, min_n:min_n + self.N]
        # plt.imshow(gray, cmap=plt.get_cmap('gray'))
        # plt.show()

    def calcGridMismatch(self, m, n, min_cost, min_m, min_n, search_region_width_m=0, search_region_width_n=0):
        # mm = m-search_region_width_m
        # nn = n+search_region_width_n
        cost = self.calculateMismatch(m, n)
        if min_cost > cost:
            min_cost = cost
            min_m = m
            min_n = n
        print(m, n, cost)
        return min_cost, min_m, min_n

    def calculateMismatch(self, m, n):
        if m + self.M > self.I or m < 0 or n + self.N > self.J or n < 0:
            return MAX_COST
        curr_test_array_slice = self.test_img_array_gray[m:m + self.M, n:n + self.N]
        return -2 * np.sum(curr_test_array_slice * self.ref_img_array_gray) + np.sum(
            self.ref_img_array_gray * self.ref_img_array_gray) + np.sum(curr_test_array_slice * curr_test_array_slice)

    def hiererchical_search_old(self):
        # level = 1
        ref_lowpass_1 = ndimage.gaussian_filter(self.ref_img_array_gray, 2)
        ref_rescaled_1 = misc.imresize(ref_lowpass_1, 0.5)
        test_lowpass_1 = ndimage.gaussian_filter(self.test_img_array_gray, 2)
        test_rescaled_1 = misc.imresize(test_lowpass_1, 0.5)

        tm = TM(ref_rescaled_1, test_rescaled_1)
        tm.exhaustive_search()
        # level = 2
        ref_lowpass_2 = ndimage.gaussian_filter(ref_rescaled_1, 2)
        ref_rescaled_2 = misc.imresize(ref_lowpass_2, 0.5)
        test_lowpass_2 = ndimage.gaussian_filter(test_rescaled_1, 2)
        test_rescaled_2 = misc.imresize(test_lowpass_2, 0.5)

        tm = TM(ref_rescaled_2, test_rescaled_2)
        min_m, min_n = tm.exhaustive_search()
        min_m, min_n = 2 * min_m, 2 * min_n
        min_cost = MAX_COST

        m = min_m - 1
        n = min_n - 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m
        n = min_n - 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m + 1
        n = min_n - 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m - 1
        n = min_n
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m
        n = min_n
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m + 1
        n = min_n
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m - 1
        n = min_n + 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m
        n = min_n + 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m + 1
        n = min_n + 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)

        min_m, min_n = 2 * min_m, 2 * min_n
        min_cost = MAX_COST

        m = min_m - 1
        n = min_n - 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m
        n = min_n - 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m + 1
        n = min_n - 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m - 1
        n = min_n
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m
        n = min_n
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m + 1
        n = min_n
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m - 1
        n = min_n + 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m
        n = min_n + 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m + 1
        n = min_n + 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        print('hiererchical res', min_m, min_n)

        gray = self.test_img_array_gray[min_m:min_m + self.M, min_n:min_n + self.N]
        plt.imshow(gray, cmap=plt.get_cmap('gray'))
        plt.show()

    def hiererchical_search(self):
        # level = 0
        center_x_0 = int(self.M / 2)
        center_y_0 = int(self.N / 2)

        # level = 1
        ref_rescaled_1 = misc.imresize(self.ref_img_array_gray, 0.5)
        test_rescaled_1 = misc.imresize(self.test_img_array_gray, 0.5)
        center_x_1 = int(center_x_0 / 2)
        center_y_1 = int(center_y_0 / 2)

        # tm = TM(ref_rescaled_1, test_rescaled_1)
        # tm.exhaustive_search()

        # level = 2
        ref_rescaled_2 = misc.imresize(ref_rescaled_1, 0.5)
        test_rescaled_2 = misc.imresize(test_rescaled_1, 0.5)
        center_x_2 = int(center_x_1 / 2)
        center_y_2 = int(center_y_1 / 2)

        tm = TM(ref_rescaled_2, test_rescaled_2)
        min_m, min_n = tm.exhaustive_search()
        min_m, min_n = 2 * min_m, 2 * min_n
        min_cost = MAX_COST

        m = min_m - 1
        n = min_n - 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m
        n = min_n - 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m + 1
        n = min_n - 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m - 1
        n = min_n
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m
        n = min_n
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m + 1
        n = min_n
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m - 1
        n = min_n + 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m
        n = min_n + 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m + 1
        n = min_n + 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)

        min_m, min_n = 2 * min_m, 2 * min_n
        min_cost = MAX_COST

        m = min_m - 1
        n = min_n - 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m
        n = min_n - 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m + 1
        n = min_n - 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m - 1
        n = min_n
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m
        n = min_n
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m + 1
        n = min_n
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m - 1
        n = min_n + 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m
        n = min_n + 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        m = min_m + 1
        n = min_n + 1
        min_cost, min_m, min_n = self.calcGridMismatch(m, n, min_cost, min_m, min_n)
        print('hiererchical res', min_m, min_n)

        gray = self.test_img_array_gray[min_m:min_m + self.M, min_n:min_n + self.N]
        plt.imshow(gray, cmap=plt.get_cmap('gray'))
        plt.show()

class TemplateMatching:
    # ref_img can be image path or can be image numpy array
    def __init__(self, ref_img, test_img):
        if type(ref_img) is str and type(test_img) is str:
            self.ref_img_array = getImageArray(ref_img)
            self.test_img_array = getImageArray(test_img)
        else:
            self.ref_img_array = ref_img
            self.test_img_array = test_img
        # print(self.ref_img_array.shape)
        # print(self.test_img_array.shape)
        if self.ref_img_array.shape[0] > self.test_img_array.shape[0] \
                or self.ref_img_array.shape[1] > self.test_img_array.shape[1]:
            raise Exception("shape error")
        self.M = self.ref_img_array.shape[0]
        self.N = self.ref_img_array.shape[1]
        self.I = self.test_img_array.shape[0]
        self.J = self.test_img_array.shape[1]

    def exhaustive_search(self):
        test_img_arr = self.test_img_array
        ref_img_arr = self.ref_img_array
        I, J = test_img_arr.shape[0], test_img_arr.shape[1]
        M, N = ref_img_arr.shape[0], ref_img_arr.shape[1]
        print(I, J, M, N)

        min_m = 0
        min_n = 0
        min_cost = MAX_COST
        for m in range(I - M + 1):
            for n in range(J - N + 1):
                cost = calculateMismatch(ref_image=ref_img_arr, test_image=test_img_arr, m=m, n=n)
                print(m, n, cost)
                if min_cost > cost:
                    min_cost = cost
                    min_m = m
                    min_n = n

        print('exhaustive res', min_m, min_n)
        m = min_m
        n = min_n
        curr_test_array_slice = test_img_arr[m:m + M, n:n + N, :]
        plt.imshow(curr_test_array_slice)
        plt.show()
        return min_m, min_n

    def two_D_logarithmic_search(self):
        None

    def hiererchical_search(self):
        None


def exhaustive_search_cor():
    ref_image = ndimage.imread("Data/ref_image.jpg", True)
    test_image = ndimage.imread("Data/test_image.png", True)
    I, J = test_image.shape[0], test_image.shape[1]
    M, N = ref_image.shape[0], ref_image.shape[1]
    min_m = 0
    min_n = 0
    max_cost = 0.0
    for m in range(I - M + 1):
        for n in range(J - N + 1):
            cost = getCrossCorrelation(test_image, ref_image, m, n)
            print(m, n, cost)
            if max_cost < cost:
                max_cost = cost
                min_m = m
                min_n = n

    print('exhaustive res', min_m, min_n)
    plt.imshow(test_image)
    plt.show()
    # gray = self.test_img_array_gray[min_m:min_m + self.M, min_n:min_n + self.N]
    # plt.imshow(gray)
    # plt.show()
    return min_m, min_n


def calcGridMismatch_cor(m, n, min_cost, min_m, min_n, ref_image, test_image):
    # mm = m-search_region_width_m
    # nn = n+search_region_width_n
    cost = getCrossCorrelation(test_image, ref_image, m, n)
    if min_cost < cost:
        min_cost = cost
        min_m = m
        min_n = n
    print(m, n, cost)
    return min_cost, min_m, min_n


def two_dim_logarithmic_search_cor():
    ref_image = ndimage.imread("Data/ref_image.jpg", True)
    test_image = ndimage.imread("Data/test_image.png", True)
    I, J = test_image.shape[0], test_image.shape[1]
    M, N = ref_image.shape[0], ref_image.shape[1]
    min_m = 0
    min_n = 0
    min_cost = 0
    center_m = int(I / 2)
    center_n = int(J / 2)
    search_region_width_m = int(max(I, J) / 2)
    search_region_width_n = int(max(I, J) / 2)
    print('center', center_m, center_n)
    print('width', search_region_width_m, search_region_width_n)

    while search_region_width_m > 1 and search_region_width_n > 1:
        m = center_m
        n = center_n
        min_cost, min_m, min_n = calcGridMismatch_cor(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                      test_image=test_image)
        m = center_m - search_region_width_m
        n = center_n - search_region_width_n
        min_cost, min_m, min_n = calcGridMismatch_cor(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                      test_image=test_image)
        m = center_m
        n = center_n - search_region_width_n
        min_cost, min_m, min_n = calcGridMismatch_cor(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                      test_image=test_image)
        m = center_m + search_region_width_m
        n = center_n - search_region_width_n
        min_cost, min_m, min_n = calcGridMismatch_cor(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                      test_image=test_image)
        m = center_m - search_region_width_m
        n = center_n
        min_cost, min_m, min_n = calcGridMismatch_cor(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                      test_image=test_image)
        m = center_m + search_region_width_m
        n = center_n
        min_cost, min_m, min_n = calcGridMismatch_cor(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                      test_image=test_image)
        m = center_m - search_region_width_m
        n = center_n + search_region_width_n
        min_cost, min_m, min_n = calcGridMismatch_cor(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                      test_image=test_image)
        m = center_m
        n = center_n + search_region_width_n
        min_cost, min_m, min_n = calcGridMismatch_cor(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                      test_image=test_image)
        m = center_m + search_region_width_m
        n = center_n + search_region_width_n
        min_cost, min_m, min_n = calcGridMismatch_cor(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                      test_image=test_image)
        if center_m == min_m and center_n == min_n:
            search_region_width_m = int(search_region_width_m / 2)
            search_region_width_n = int(search_region_width_n / 2)
        center_m = min_m
        center_n = min_n
        print('center', center_m, center_n)
        print('width', search_region_width_m, search_region_width_n)
        print()
    print('res', min_m, min_n)
    # gray = self.test_img_array_gray[min_m:min_m + self.M, min_n:min_n + self.N]
    # plt.imshow(gray, cmap=plt.get_cmap('gray'))
    # plt.show()


def getCrossCorrelation(sourceImage, templateImage, m, n):
    M, N = templateImage.shape[0], templateImage.shape[1]
    I, J = sourceImage.shape[0], sourceImage.shape[1]
    if m + M > I or m < 0 or n + N > J or n < 0:
        return 0.0
    M, N = templateImage.shape[0], templateImage.shape[1]
    cross_correlation = 0
    sum_source = 0
    sum_template = 0
    cross_correlation_normalized = 0
    for i in range(m, m + M):
        for j in range(n, n + N):
            cross_correlation += (int(sourceImage[i, j]) * int(templateImage[i - m, j - n]))
            sum_source += (int(sourceImage[i, j]) ** 2)
            sum_template += (int(templateImage[i - m, j - n])) ** 2

        cross_correlation_normalized = cross_correlation / math.sqrt(sum_source * sum_template)
    return cross_correlation_normalized


def calculateMismatch(ref_image, test_image, m, n):
    M, N = ref_image.shape[0], ref_image.shape[1]
    I, J = test_image.shape[0], test_image.shape[1]
    # print(I, J, M, N)
    if m + M > I or m < 0 or n + N > J or n < 0:
        # print("?")
        return MAX_COST
    curr_test_array_slice = test_image[m:m + M, n:n + N, :]
    # print(ref_image.shape, curr_test_array_slice.shape)
    # print(np.sum(curr_test_array_slice * ref_image) , np.sum(
    #     ref_image * ref_image) , np.sum(curr_test_array_slice * curr_test_array_slice))
    return -2 * np.sum(curr_test_array_slice * ref_image) + np.sum(
        ref_image * ref_image) + np.sum(curr_test_array_slice * curr_test_array_slice)


def exhaustive_search_cor_my():
    ref_image = np.array(Image.open("Data/ref_image.jpg"), dtype=int)
    im = Image.open("Data/test_image.png")
    test_image = Image.new("RGB", im.size, (255, 255, 255))
    test_image.paste(im, (0, 0), im)
    test_image = np.array(test_image, dtype=int)
    I, J = test_image.shape[0], test_image.shape[1]
    M, N = ref_image.shape[0], ref_image.shape[1]
    print(I, J, M, N)

    min_m = 0
    min_n = 0
    min_cost = MAX_COST
    for m in range(I - M + 1):
        for n in range(J - N + 1):
            cost = calculateMismatch(ref_image=ref_image, test_image=test_image, m=m, n=n)
            print(m, n, cost)
            if min_cost > cost:
                min_cost = cost
                min_m = m
                min_n = n

    print('exhaustive res', min_m, min_n)
    m = min_m
    n = min_n
    curr_test_array_slice = test_image[m:m + M, n:n + N, :]
    plt.imshow(curr_test_array_slice)
    plt.show()
    return min_m, min_n


def calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image, test_image):
    # mm = m-search_region_width_m
    # nn = n+search_region_width_n
    cost = calculateMismatch(ref_image, test_image, m, n)
    if min_cost > cost:
        min_cost = cost
        min_m = m
        min_n = n
    print(m, n, cost)
    return min_cost, min_m, min_n


def two_dim_logarithmic_search_cor_my():
    ref_image = np.array(Image.open("Data/ref_image.jpg"), dtype=int)
    im = Image.open("Data/test_image.png")
    test_image = Image.new("RGB", im.size, (255, 255, 255))
    test_image.paste(im, (0, 0), im)
    test_image = np.array(test_image, dtype=int)
    # test_image = np.array(Image.open("Data/test_2.jpg"), dtype=int)
    I, J = test_image.shape[0], test_image.shape[1]
    M, N = ref_image.shape[0], ref_image.shape[1]
    print(I, J, M, N)

    min_m = 0
    min_n = 0
    min_cost = MAX_COST
    center_m = int(I / 2)
    center_n = int(J / 2)
    search_region_width_m = math.ceil(I / 4)  # int(max(I,J)/2)
    search_region_width_n = math.ceil(J / 4)  # int(max(I,J)/2)
    print('center', center_m, center_n)
    print('width', search_region_width_m, search_region_width_n)

    while search_region_width_m > 1 and search_region_width_n > 1:
        m = center_m
        n = center_n
        min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                     test_image=test_image)
        m = center_m - search_region_width_m
        n = center_n - search_region_width_n
        min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                     test_image=test_image)
        m = center_m
        n = center_n - search_region_width_n
        min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                     test_image=test_image)
        m = center_m + search_region_width_m
        n = center_n - search_region_width_n
        min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                     test_image=test_image)
        m = center_m - search_region_width_m
        n = center_n
        min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                     test_image=test_image)
        m = center_m + search_region_width_m
        n = center_n
        min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                     test_image=test_image)
        m = center_m - search_region_width_m
        n = center_n + search_region_width_n
        min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                     test_image=test_image)
        m = center_m
        n = center_n + search_region_width_n
        min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                     test_image=test_image)
        m = center_m + search_region_width_m
        n = center_n + search_region_width_n
        min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                     test_image=test_image)
        if center_m == min_m and center_n == min_n:
            search_region_width_m = int(search_region_width_m / 2)
            search_region_width_n = int(search_region_width_n / 2)
        center_m = min_m
        center_n = min_n
        print('center', center_m, center_n)
        print('width', search_region_width_m, search_region_width_n)
        print()
    print('res', min_m, min_n)
    m = min_m
    n = min_n
    curr_test_array_slice = test_image[m:m + M, n:n + N, :]
    plt.imshow(curr_test_array_slice)
    plt.show()


def two_dim_logarithmic_search_my(ref_image, test_image):
    I, J = test_image.shape[0], test_image.shape[1]
    M, N = ref_image.shape[0], ref_image.shape[1]
    print(I, J, M, N)

    min_m = 0
    min_n = 0
    min_cost = MAX_COST
    center_m = int(I / 2)
    center_n = int(J / 2)
    search_region_width_m = math.ceil(I / 4)  # int(max(I,J)/2)
    search_region_width_n = math.ceil(J / 4)  # int(max(I,J)/2)
    print('center', center_m, center_n)
    print('width', search_region_width_m, search_region_width_n)

    while search_region_width_m > 1 and search_region_width_n > 1:
        m = center_m
        n = center_n
        min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                     test_image=test_image)
        m = center_m - search_region_width_m
        n = center_n - search_region_width_n
        min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                     test_image=test_image)
        m = center_m
        n = center_n - search_region_width_n
        min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                     test_image=test_image)
        m = center_m + search_region_width_m
        n = center_n - search_region_width_n
        min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                     test_image=test_image)
        m = center_m - search_region_width_m
        n = center_n
        min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                     test_image=test_image)
        m = center_m + search_region_width_m
        n = center_n
        min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                     test_image=test_image)
        m = center_m - search_region_width_m
        n = center_n + search_region_width_n
        min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                     test_image=test_image)
        m = center_m
        n = center_n + search_region_width_n
        min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                     test_image=test_image)
        m = center_m + search_region_width_m
        n = center_n + search_region_width_n
        min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                     test_image=test_image)
        if center_m == min_m and center_n == min_n:
            search_region_width_m = int(search_region_width_m / 2)
            search_region_width_n = int(search_region_width_n / 2)
        center_m = min_m
        center_n = min_n
        print('center', center_m, center_n)
        print('width', search_region_width_m, search_region_width_n)
        print()
    print('res', min_m, min_n)
    m = min_m
    n = min_n
    curr_test_array_slice = test_image[m:m + M, n:n + N, :]
    plt.imshow(curr_test_array_slice)
    plt.show()
    return min_m, min_n


def exhaustive_search_my(ref_image, test_image):
    I, J = test_image.shape[0], test_image.shape[1]
    M, N = ref_image.shape[0], ref_image.shape[1]
    print(I, J, M, N)

    min_m = 0
    min_n = 0
    min_cost = MAX_COST
    for m in range(I - M + 1):
        for n in range(J - N + 1):
            cost = calculateMismatch(ref_image=ref_image, test_image=test_image, m=m, n=n)
            print(m, n, cost)
            if min_cost > cost:
                min_cost = cost
                min_m = m
                min_n = n

    print('exhaustive res', min_m, min_n)
    m = min_m
    n = min_n
    curr_test_array_slice = test_image[m:m + M, n:n + N, :]
    plt.imshow(curr_test_array_slice)
    plt.show()
    return min_m, min_n


def hiererchical_search_my():
    ref_image = np.array(Image.open("Data/ref_image.jpg"), dtype=int)
    im = Image.open("Data/test_image.png")
    test_image = Image.new("RGB", im.size, (255, 255, 255))
    test_image.paste(im, (0, 0), im)
    test_image = np.array(test_image, dtype=int)
    # test_image = np.array(Image.open("Data/test_2.jpg"), dtype=int)
    I, J = test_image.shape[0], test_image.shape[1]
    M, N = ref_image.shape[0], ref_image.shape[1]
    print(I, J, M, N)
    # level = 0
    center_x_0 = int(M / 2)
    center_y_0 = int(N / 2)

    # level = 1
    ref_rescaled_1 = misc.imresize(ref_image, 0.5)
    test_rescaled_1 = misc.imresize(test_image, 0.5)
    center_x_1 = int(center_x_0 / 2)
    center_y_1 = int(center_y_0 / 2)

    # tm = TM(ref_rescaled_1, test_rescaled_1)
    # tm.exhaustive_search()

    # level = 2
    ref_rescaled_2 = misc.imresize(ref_rescaled_1, 0.5)
    test_rescaled_2 = misc.imresize(test_rescaled_1, 0.5)
    center_x_2 = int(center_x_1 / 2)
    center_y_2 = int(center_y_1 / 2)

    min_m, min_n = two_dim_logarithmic_search_my(ref_rescaled_2, test_rescaled_2)
    min_m, min_n = 2 * min_m, 2 * min_n
    min_cost = MAX_COST

    m = min_m - 1
    n = min_n - 1
    min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                 test_image=test_image)
    m = min_m
    n = min_n - 1
    min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                 test_image=test_image)
    m = min_m + 1
    n = min_n - 1
    min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                 test_image=test_image)
    m = min_m - 1
    n = min_n
    min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                 test_image=test_image)
    m = min_m
    n = min_n
    min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                 test_image=test_image)
    m = min_m + 1
    n = min_n
    min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                 test_image=test_image)
    m = min_m - 1
    n = min_n + 1
    min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                 test_image=test_image)
    m = min_m
    n = min_n + 1
    min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                 test_image=test_image)
    m = min_m + 1
    n = min_n + 1
    min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                 test_image=test_image)

    min_m, min_n = 2 * min_m, 2 * min_n
    min_cost = MAX_COST

    m = min_m - 1
    n = min_n - 1
    min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                 test_image=test_image)
    m = min_m
    n = min_n - 1
    min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                 test_image=test_image)
    m = min_m + 1
    n = min_n - 1
    min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                 test_image=test_image)
    m = min_m - 1
    n = min_n
    min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                 test_image=test_image)
    m = min_m
    n = min_n
    min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                 test_image=test_image)
    m = min_m + 1
    n = min_n
    min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                 test_image=test_image)
    m = min_m - 1
    n = min_n + 1
    min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                 test_image=test_image)
    m = min_m
    n = min_n + 1
    min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                 test_image=test_image)
    m = min_m + 1
    n = min_n + 1
    min_cost, min_m, min_n = calcGridMismatch_my(m, n, min_cost, min_m, min_n, ref_image=ref_image,
                                                 test_image=test_image)
    print('hiererchical res', min_m, min_n)

    m = min_m
    n = min_n
    curr_test_array_slice = test_image[m:m + M, n:n + N, :]
    plt.imshow(curr_test_array_slice)
    plt.show()


# hiererchical_search_my()
# two_dim_logarithmic_search_cor()
# im = Image.open("Data/test_image.png")
# print(np.array(im).shape)
# exhaustive_search_cor_my()
# two_dim_logarithmic_search_cor_my()
tm = TM("Data/ref_image.jpg", "Data/test_image.png")
# tm.two_dim_logarithmic_search()
tm.hiererchical_search()
