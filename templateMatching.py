import math
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from scipy import ndimage, misc
import timeit

MAX_COST = 2 ** 64 - 1


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
        # plt.imshow(self.test_img_array_gray, cmap=plt.get_cmap('gray'))
        # plt.show()
        # gray = self.test_img_array_gray[min_m:min_m + self.M, min_n:min_n + self.N]
        # plt.imshow(gray, cmap=plt.get_cmap('gray'))
        # plt.show()
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
        print('logarithmic res', min_m, min_n)
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

        # gray = self.test_img_array_gray[min_m:min_m + self.M, min_n:min_n + self.N]
        # plt.imshow(gray, cmap=plt.get_cmap('gray'))
        # plt.show()


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def get_image_array(img_path):
    if ".jpg" in img_path:
        return np.array(Image.open(img_path), dtype=np.int64)
    elif ".png" in img_path:
        im = Image.open(img_path)
        image = Image.new("RGB", im.size, (255, 255, 255))
        image.paste(im, (0, 0), im)
        return np.array(image, dtype=np.int64)


def draw_result(img_arr, m, n, M, N):
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(img_arr)
    # Create a Rectangle patch
    rect = patches.Rectangle((n, m), N, M, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()


def calculate_mismatch(ref_image, test_image, m, n):
    M, N = ref_image.shape[0], ref_image.shape[1]
    I, J = test_image.shape[0], test_image.shape[1]
    # print(I, J, M, N)
    # check if the ref image goes out of bound of the original image or not
    if m + M > I or m < 0 or n + N > J or n < 0:
        # print("?")
        return MAX_COST
    curr_test_array_slice = test_image[m:m + M, n:n + N, :]
    # print(ref_image.shape, curr_test_array_slice.shape)
    # print(np.sum(curr_test_array_slice * ref_image) , np.sum(
    #     ref_image * ref_image) , np.sum(curr_test_array_slice * curr_test_array_slice))
    return -2 * np.sum(curr_test_array_slice * ref_image) \
           + np.sum(ref_image * ref_image) + np.sum(curr_test_array_slice * curr_test_array_slice)


def calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image, test_image):
    # mm = m-search_region_width_m
    # nn = n+search_region_width_n
    cost = calculate_mismatch(ref_image, test_image, m, n)
    if min_cost > cost:
        min_cost = cost
        min_m = m
        min_n = n
    # print(m, n, cost)
    return min_cost, min_m, min_n


def normalize_image_array(image):
    return image / (image.max() / 255.0)


def hierarchical_search_recursive(ref_img_arr, test_img_arr, curr_level, total_level):
    if curr_level == total_level:
        misc.imsave('ref_level_special.jpg', ref_img_arr)
        misc.imsave('test_level_special.jpg', test_img_arr)
        # ref_img_arr = np.array(misc.toimage(ref_img_arr, channel_axis=2))
        # test_img_arr = np.array(misc.toimage(test_img_arr, channel_axis=2))
        template_matching = TemplateMatching('ref_level_special.jpg', 'test_level_special.jpg')
        # template_matching = TemplateMatching(ref_img_arr, test_img_arr)
        min_m, min_n = template_matching.exhaustive_search()
        os.remove('ref_level_special.jpg')
        os.remove('test_level_special.jpg')
        return min_m, min_n
    else:
        # low pass filtering and then rescale
        ref_lowpass_filtered = ndimage.gaussian_filter(misc.toimage(ref_img_arr, channel_axis=2), sigma=(2, 2, 0))
        ref_rescaled = misc.imresize(ref_lowpass_filtered, 0.5)
        test_lowpass_filtered = ndimage.gaussian_filter(misc.toimage(test_img_arr, channel_axis=2), sigma=(2, 2, 0))
        test_rescaled = misc.imresize(test_lowpass_filtered, 0.5)

        # rescaling without low pass filtering
        # ref_rescaled = misc.imresize(ref_img_arr, 0.5)
        # test_rescaled = misc.imresize(test_img_arr, 0.5)

        # save image
        # misc.imsave('ref_level'+repr(curr_level)+'.jpg',ref_rescaled)
        # misc.imsave('test_level'+repr(curr_level)+'.jpg',test_rescaled)

        m, n = hierarchical_search_recursive(ref_rescaled, test_rescaled, curr_level + 1, total_level)
        print('level', curr_level, ': ', m, n)

        min_m, min_n = 2 * m, 2 * n
        min_cost = MAX_COST

        m = min_m - 1
        n = min_n - 1
        min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                    test_image=test_img_arr)
        m = min_m
        n = min_n - 1
        min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                    test_image=test_img_arr)
        m = min_m + 1
        n = min_n - 1
        min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                    test_image=test_img_arr)
        m = min_m - 1
        n = min_n
        min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                    test_image=test_img_arr)
        m = min_m
        n = min_n
        min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                    test_image=test_img_arr)
        m = min_m + 1
        n = min_n
        min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                    test_image=test_img_arr)
        m = min_m - 1
        n = min_n + 1
        min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                    test_image=test_img_arr)
        m = min_m
        n = min_n + 1
        min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                    test_image=test_img_arr)
        m = min_m + 1
        n = min_n + 1
        min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                    test_image=test_img_arr)

        return min_m, min_n


class TemplateMatching:
    # ref_img can be image path or can be image numpy array
    def __init__(self, ref_img, test_img, inspect=False):
        self.inspect = inspect
        if type(ref_img) is str and type(test_img) is str:
            self.ref_img_array = get_image_array(ref_img)
            self.test_img_array = get_image_array(test_img)
        else:
            self.ref_img_array = ref_img
            self.test_img_array = test_img
        # print(self.ref_img_array.shape)
        # print(self.test_img_array.shape)
        if self.ref_img_array.shape[0] > self.test_img_array.shape[0] \
                or self.ref_img_array.shape[1] > self.test_img_array.shape[1]:
            raise Exception("shape error")
            # self.M = self.ref_img_array.shape[0]
            # self.N = self.ref_img_array.shape[1]
            # self.I = self.test_img_array.shape[0]
            # self.J = self.test_img_array.shape[1]

    def exhaustive_search(self):
        test_img_arr = self.test_img_array
        ref_img_arr = self.ref_img_array
        I, J = test_img_arr.shape[0], test_img_arr.shape[1]
        M, N = ref_img_arr.shape[0], ref_img_arr.shape[1]
        # print("I,J,M,N ", I, J, M, N)

        min_m = 0
        min_n = 0
        min_cost = MAX_COST
        for m in range(I - M + 1):
            for n in range(J - N + 1):
                cost = calculate_mismatch(ref_image=ref_img_arr, test_image=test_img_arr, m=m, n=n)
                # print(m, n, cost)
                if min_cost > cost:
                    min_cost = cost
                    min_m = m
                    min_n = n

        print('exhaustive res', min_m, min_n)
        m = min_m
        n = min_n

        if self.inspect:
            draw_result(test_img_arr, m, n, M, N)
            curr_test_array_slice = test_img_arr[m:m + M, n:n + N, :]
            plt.imshow(curr_test_array_slice)
            plt.show()
        return min_m, min_n

    def two_D_logarithmic_search(self):
        test_img_arr = self.test_img_array
        ref_img_arr = self.ref_img_array
        I, J = test_img_arr.shape[0], test_img_arr.shape[1]
        M, N = ref_img_arr.shape[0], ref_img_arr.shape[1]
        # print("I,J,M,N ", I, J, M, N)

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
            min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                        test_image=test_img_arr)
            m = center_m - search_region_width_m
            n = center_n - search_region_width_n
            min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                        test_image=test_img_arr)
            m = center_m
            n = center_n - search_region_width_n
            min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                        test_image=test_img_arr)
            m = center_m + search_region_width_m
            n = center_n - search_region_width_n
            min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                        test_image=test_img_arr)
            m = center_m - search_region_width_m
            n = center_n
            min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                        test_image=test_img_arr)
            m = center_m + search_region_width_m
            n = center_n
            min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                        test_image=test_img_arr)
            m = center_m - search_region_width_m
            n = center_n + search_region_width_n
            min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                        test_image=test_img_arr)
            m = center_m
            n = center_n + search_region_width_n
            min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                        test_image=test_img_arr)
            m = center_m + search_region_width_m
            n = center_n + search_region_width_n
            min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                        test_image=test_img_arr)
            # if center_m == min_m and center_n == min_n:
            search_region_width_m = int(search_region_width_m / 2)
            search_region_width_n = int(search_region_width_n / 2)
            center_m = min_m
            center_n = min_n
            # print('center', center_m, center_n)
            # print('width', search_region_width_m, search_region_width_n)
            # print()
        print('2D logarithmic search res', min_m, min_n)
        m = min_m
        n = min_n

        if self.inspect:
            draw_result(test_img_arr, m, n, M, N)
            curr_test_array_slice = test_img_arr[m:m + M, n:n + N, :]
            plt.imshow(curr_test_array_slice)
            plt.show()
        return m, n

    def two_D_logarithmic_search_square(self):
        test_img_arr = self.test_img_array
        ref_img_arr = self.ref_img_array
        I, J = test_img_arr.shape[0], test_img_arr.shape[1]
        M, N = ref_img_arr.shape[0], ref_img_arr.shape[1]
        # print("I,J,M,N ", I, J, M, N)

        min_m = 0
        min_n = 0
        min_cost = MAX_COST
        center_m = int(I / 2)
        center_n = center_m
        search_region_width_m = math.ceil(I / 4)  # int(max(I,J)/2)
        search_region_width_n = search_region_width_m
        print('center', center_m, center_n)
        print('width', search_region_width_m, search_region_width_n)

        while search_region_width_m > 1 and search_region_width_n > 1:
            m = center_m
            n = center_n
            min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                        test_image=test_img_arr)
            m = center_m - search_region_width_m
            n = center_n - search_region_width_n
            min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                        test_image=test_img_arr)
            m = center_m
            n = center_n - search_region_width_n
            min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                        test_image=test_img_arr)
            m = center_m + search_region_width_m
            n = center_n - search_region_width_n
            min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                        test_image=test_img_arr)
            m = center_m - search_region_width_m
            n = center_n
            min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                        test_image=test_img_arr)
            m = center_m + search_region_width_m
            n = center_n
            min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                        test_image=test_img_arr)
            m = center_m - search_region_width_m
            n = center_n + search_region_width_n
            min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                        test_image=test_img_arr)
            m = center_m
            n = center_n + search_region_width_n
            min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                        test_image=test_img_arr)
            m = center_m + search_region_width_m
            n = center_n + search_region_width_n
            min_cost, min_m, min_n = calc_grid_mismatch(m, n, min_cost, min_m, min_n, ref_image=ref_img_arr,
                                                        test_image=test_img_arr)
            # if center_m == min_m and center_n == min_n:
            search_region_width_m = int(search_region_width_m / 2)
            search_region_width_n = search_region_width_m
            center_m = min_m
            center_n = min_n
            # print('center', center_m, center_n)
            # print('width', search_region_width_m, search_region_width_n)
            # print()
        print('2D logarithmic search square res', min_m, min_n)
        m = min_m
        n = min_n

        if self.inspect:
            draw_result(test_img_arr, m, n, M, N)
            curr_test_array_slice = test_img_arr[m:m + M, n:n + N, :]
            plt.imshow(curr_test_array_slice)
            plt.show()
        return m, n

    def hierarchical_search(self, ref_img_arr=None, test_img_arr=None):
        if ref_img_arr is None and test_img_arr is None:
            test_img_arr = self.test_img_array
            ref_img_arr = self.ref_img_array
        I, J = test_img_arr.shape[0], test_img_arr.shape[1]
        M, N = ref_img_arr.shape[0], ref_img_arr.shape[1]
        # print("I,J,M,N ", I, J, M, N)

        m, n = hierarchical_search_recursive(ref_img_arr, test_img_arr, 0, 2)
        print('hierarchical res', m, n)

        if self.inspect:
            draw_result(test_img_arr, m, n, M, N)
            curr_test_array_slice = test_img_arr[m:m + M, n:n + N, :]
            plt.imshow(curr_test_array_slice)
            plt.show()
        return m, n


test = "Data/test_image.png"
ref = "Data/ref_image.jpg"


def experiment_exhaustive():
    tm = TemplateMatching(ref, test)
    tm.exhaustive_search()


def experiment_logarithmic():
    tm = TemplateMatching(ref, test)
    tm.two_D_logarithmic_search()


def experiment_hierarchical():
    tm = TemplateMatching(ref, test)
    tm.hierarchical_search()


def exhaustive_search_time():
    SETUP_CODE = '''
from __main__ import experiment_exhaustive
from random import randint'''

    TEST_CODE = '''
experiment_exhaustive()'''
    print('EXHAUSTIVE SEARCH')
    # timeit.repeat statement
    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=3,
                          number=1)

    # priniting minimum exec. time
    print('required time: {}'.format(min(times)))
    return sum(times) / len(times)


def logarithmic_search_time():
    SETUP_CODE = '''
from __main__ import experiment_logarithmic
from random import randint'''

    TEST_CODE = '''
experiment_logarithmic()'''
    print('LOGARITHMIC SEARCH')
    # timeit.repeat statement
    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=3,
                          number=1)

    # priniting minimum exec. time
    print('required time: {}'.format(min(times)))
    return sum(times) / len(times)


def hierarchical_search_time():
    SETUP_CODE = '''
from __main__ import experiment_hierarchical
from random import randint'''

    TEST_CODE = '''
experiment_hierarchical()'''
    print('HIERARCHICAL SEARCH')
    # timeit.repeat statement
    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=3,
                          number=1)

    # priniting minimum exec. time
    print('required time: {}'.format(min(times)))
    return sum(times) / len(times)


def experiment_runtime():
    times = [exhaustive_search_time(), logarithmic_search_time(), hierarchical_search_time()]
    print()
    print('exhaustive search time ', times[0], ' sec')
    print('2D logarithmic search time ', times[1], ' sec')
    print('hierarchical search time ', times[2], ' sec')


# experiment_runtime()

# test = "Data/test_image.png"
# ref = "Data/ref_image.jpg"
test2 = "Data/test_2.jpg"
ref2 = "Data/ref_2.jpg"
tm = TemplateMatching(ref2, test2)
tm.hierarchical_search()
