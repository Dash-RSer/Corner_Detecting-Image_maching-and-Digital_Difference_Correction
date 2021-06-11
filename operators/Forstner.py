# -*- coding: utf-8 -*-
# @author: Dasheng Fan
# @email: fandasheng1999@163.com

"""
This code is used for build a Forstner operater for extrating
corners of a image.
"""

import numpy as np
np.set_printoptions(threshold=np.inf)
import sys
import os
import numba as nb
from matplotlib import pyplot as plt

from utils.gray_processing import gray_processing
from utils.display import display

class Forstner(object):  
    """
    Forstner.

    Forstner is a operator for corner extracting.The paper
    is:"A fast operator for detection and precise location 
    of distinct points, corners and circular features"

    """
    
    def __init__(self, kernel_size = 5,
                 threshold_for_q = 0.5, 
                 f = 0.5, 
                 c = 5, 
                 maximum_kernel_size = 7, 
                 threshold_for_pre_select = 20):
        """
         Args:
            kernel_size: the window size to calculate the covariance matrix
            threshold_for_q: The threshold for culculate intrests.
            f: Threshold parameter.
            maximum_kernel_size: The window size for choosing the maximum.
        """
        self._threshold_q = threshold_for_q
        self._f = f
        self._kernel_size = kernel_size
        self._c = c
        self._maximum_kernel_size = maximum_kernel_size
        self._t_for_pre_s = threshold_for_pre_select
    
    def detect(self, image):


        if len(image.shape) != 2:
           image = gray_processing(image)
        
        height = image.shape[0]
        width = image.shape[1]

        cut_size = np.floor(self._kernel_size/2).astype(np.int)

        # for every pixel
        w_matrix = np.zeros((height, width), dtype = np.float32)
        q_matrix = np.zeros((height, width), dtype = np.float32)
        w_list = []

        flags = np.zeros((height, width), dtype = np.float32)



        # pre_select points
        print("start pre_select.")

        @nb.jit(nopython = True)        
        def pre_select(image, height, width, cut_size, flags, threshold):
            for row in range(cut_size, height-cut_size):
                for col in range(cut_size, width - cut_size):
                    diff = np.zeros(4, np.float32)
                    diff[0] = image[row, col] - image[row, col+1]
                    diff[1] = image[row, col] - image[row+1, col]
                    diff[2] = image[row, col] - image[row, col-1]
                    diff[3] = image[row, col] - image[row-1, col]
                    diff = np.abs(diff)

                    if np.median(diff) >= threshold:
                        flags[row, col] = 1
            
            return flags
        
        flags = pre_select(image, height, width, cut_size, flags, self._t_for_pre_s)
        
        print("pre_select over.")
        print("pre-selected points:", len(np.where(flags == 1)[0]))

        print("start:", end ="")
        for row in range(cut_size, height - cut_size):
            if row % 200 == 0:
                print(":", end = "")
            for col in range(cut_size, width - cut_size):
                if flags[row, col] == 1:
                    # N for a window in (kernel_size, kernel_size)    
                    sum_gu_2 = 0
                    sum_gv_2 = 0
                    sum_gu_gv = 0
                    
                    N = np.zeros((2,2), dtype = np.float)
                    
                    
                    for var_x in range(-1*cut_size, cut_size):
                        for var_y in range(-1*cut_size, cut_size):
                            
                            sum_gu_2 += (image[row+var_y+1, col+var_x+1] - image[row+var_y, col+var_x])**2
                            sum_gv_2 += (image[row+var_y+1, col+var_x] - image[row+var_y, col+var_x+1])**2
                            sum_gu_gv += (image[row+var_y+1, col+var_x+1] - image[row+var_y, col+var_x])*\
                                (image[row+var_y+1, col+var_x] - image[row+var_y, col+var_x+1])
                            
                            N[0, 0] = sum_gu_2
                            N[1, 1] = sum_gv_2
                            N[0, 1] = sum_gu_gv
                            N[1, 0] = sum_gu_gv

                    q = 4*np.linalg.det(N)/((np.trace(N))**2)
                    w = np.linalg.det(N)/np.trace(N)

                    w_matrix[row, col] = w
                    q_matrix[row, col] = q
                    w_list.append(w)
        
        # find the median and mean of all weight
        w = np.array(w, dtype = np.float16)
        w_mean = np.mean(w_list)
        w_median = np.median(w_list)

        # here we didn't need to start from cut_size
        # because if w or q is 0 (as we initialized),
        # then it will less than the threshold.
        
        for row in range(height):
            if row % 200 == 0:
                print(":", end = "")
            for col in range(width):
                if flags[row, col] != 0:
                    if w_matrix[row, col] < self._f*w_median or \
                        w_matrix[row, col] < self._c*w_mean or \
                        q_matrix[row, col] < self._threshold_q:
                        flags[row, col] = 0

        cut_size_max = np.floor(self._maximum_kernel_size/2).astype(np.int8)

        for row in range(cut_size_max, height - cut_size_max):
            if row % 200 == 0:
                print(":", end = "")
            for col in range(cut_size_max, width-cut_size_max):
                if flags[row, col] != 0:
                    window = \
                        np.zeros((self._maximum_kernel_size, self._maximum_kernel_size),\
                            dtype = np.float)
                    window[:, :] = \
                        w_matrix[row-cut_size_max:row+cut_size_max+1, col-cut_size_max:col+cut_size_max+1]

                    if w_matrix[row, col] != np.max(window):
                        flags[row, col] = 0
        print("end.")

        print("corner points:", len(np.where(flags==1)[0]))

        return flags 

    def __call__(self, image):
        self.detect(image)


def main():
    forstner = Forstner()

if __name__ == "__main__":
    main()




            




                



