# -*- coding: utf-8 -*-
# @author: Dasheng Fan
# @email: fandasheng1999@163.com

"""
This code is used for detect the corner points
"""
import numpy as np
import numba

from operators.diptools import convolution
from utils.gray_processing import gray_processing
from utils.display import display
from operators.diptools import generate_Guassian_template

class Harris(object):
    """
    For conner detection.
    """

    def __init__(self, 
                Guassian_kernel_size = 3, 
                Guassian_sigma = 1, 
                k = 0.04, 
                kernel_size = 3,
                maximum_kernel_size = 7,
                cut_ratio = 99.98):
        
        self._G_kernel_size = Guassian_kernel_size
        self._G_sigma = Guassian_sigma
        self._k = k
        self._kernel_size = kernel_size
        self._m_kernel_size = maximum_kernel_size
        self._G_sigma = Guassian_sigma
        self._cut_ratio = cut_ratio

    
    def detect(self, image):

        if len(image.shape) != 2:
            image = gray_processing(image)
        
        # calculate Ix, Iy and Ixy for following processing
        @numba.jit(nopython = True)
        def difference(image):            
            height, width = image.shape
            
            Ix = np.zeros((height, width), dtype = np.float32)
            Iy = np.zeros((height, width), dtype = np.float32)

            for row in range(1, height-1):
                for col in range(1, width - 1):
                    Ix[row, col] = image[row, col+1] - image[row, col-1]
                    Iy[row, col] = image[row+1, col] -image[row - 1, col]

            Ix2 = np.multiply(Ix, Ix)
            Iy2 = np.multiply(Iy, Iy)
            Ixy = np.multiply(Ix, Iy)

            return Ix2, Iy2, Ixy

        print("Harris detecting...")
        Ix2, Iy2, Ixy = difference(image)

        # do the gaussian filter, and the code of gaussian is in './diptools'.
        def filter(image):

            template = generate_Guassian_template(self._G_kernel_size, self._G_sigma)
            result = convolution(image, template)

            return result

        A = filter(Ix2)
        B = filter(Iy2)
        C = filter(Ixy)
        k = self._k
        
        
        
        # generate R for every pixel
        @numba.jit(nopython = True)       
        def genRallpixel(A, B, C, k):

            height, width = A.shape
            R = np.zeros(A.shape, dtype = np.float32)

            for row in range(height):
                for col in range(width):
                    a = A[row, col]
                    b = B[row, col]
                    c = C[row, col]

                    det_M = np.multiply(a, b) - np.power(c, 2)
                    trace_M = np.add(a, b)
                    
                    var = det_M - np.multiply(k, np.power(trace_M, 2))
                    R[row, col] = var
            
            return R

        R = genRallpixel(A, B, C, k)
        height, width = R.shape
        CUT_RATIO = self._cut_ratio
        threshold = np.percentile(R, CUT_RATIO)

        # non maximum suppress in a kerbel of 7*7
        @numba.jit(nopython = True)
        def nonemaximum(R, kernel_size, threshold):

            halfsize = np.int8(np.floor(kernel_size/2))
            height, width = R.shape
            flags = np.zeros(R.shape, dtype = np.float32)

            for row in range(halfsize, height-halfsize):
                for col in range(halfsize, width-halfsize):
                    window = R[row-halfsize:row+halfsize+1, col-halfsize:col+halfsize+1]
                    if R[row, col] == np.max(window) and R[row, col] >= threshold:
                        flags[row, col] = 1
            
            return flags

        mask = nonemaximum(R, self._m_kernel_size, threshold)
        print("finish.")
        return mask
