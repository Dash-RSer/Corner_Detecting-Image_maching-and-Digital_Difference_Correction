
# -*- coding: utf-8 -*-
# @author: Dasheng Fan
# @email: fandasheng1999@163.com

import numpy as np
import numba
import sys

sys.path.append('../')
from utils.gray_processing import gray_processing

class RobertsGradient(object):
    """
    Roberts is a kind of gradient of the image.

    """

    def __init__(self):
        """
        """
        
    def calculate(self, image):
        """
        Culculate the Roberts gradient gx and gy.

        Args:
            image: image to input.

        Return:
            gx, gy
        """

        if len(image.shape) != 2:
            image = gray_processing(image)

        @numba.jit(nopython = True)
        def run(image):

            height, width = image.shape

            gradient_gy_img = np.zeros((height, width), dtype = np.float32)
            gradient_gx_img = np.zeros((height, width), dtype = np.float32)

            for row in range(1, height-1):
                for col in range(1, width-1):
                    gradient_gx_img[row, col] = image[row, col] - image[row+1, col+1]
                    gradient_gy_img[row, col] = image[row+1, col] - image[row, col+1]

            return gradient_gx_img, gradient_gy_img

        gx, gy = run(image)

        return gx, gy


class Gradient(object):

    def __init__(self):
        pass
    
    
    def calculate(self, image):

        @numba.jit(nopython=True)
        def run(image):
            height, width = image.shape

            gx = np.zeros((height, width), dtype = np.float32)
            gy = np.zeros((height, width), dtype = np.float32)

            for row in range(1, height-1):
                for col in range(1, width-1):
                    gx[row, col] = image[row, col+1] - image[row, col]
                    gy[row, col] = image[row+1, col] - image[row, col]

            return gx, gy

        gx, gy = run(image)
        return gx, gy



        


