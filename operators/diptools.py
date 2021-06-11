# -*- coding: utf-8 -*-
# @author: Dasheng Fan
# @email: fandasheng1999@163.com

"""
Some widely used operators in DIP
"""
import numpy as np
import numba
import sys
import cv2 as cv

def convolution(image, kernel):
    """
    This function is used for convolution.
    Args:
        image: image input.
        kerbel: the filter.
    """

    kernel_height, kernel_width = kernel.shape
    height, width = image.shape
    kernel_size = kernel_height

    half_size = np.floor(kernel_size/2)

    

    @numba.jit(nopython = True)
    def conv(image, kernel, half_size, height, width):

        result = np.zeros((height, width), dtype = np.float32)
        for row in range(half_size, height - half_size):
            for col in range(half_size, width - half_size):
                sum_var = 0
                for v_row in range(-1*half_size, half_size+1):
                    for v_col in range(-1*half_size, half_size+1):
                        sum_var = sum_var + image[row+v_row, col+v_col] * kernel[v_row, v_col]

                result[row, col] = sum_var

        return result

    result = conv(image, kernel, half_size, height, width)
    return result

def generate_Guassian_template(kernel_size = 3, sigma = 1):
                
    template = np.zeros((kernel_size, kernel_size), \
                    dtype = np.float32)
    halfsize = np.floor(kernel_size/2).astype(np.int16)

    @numba.jit(nopython = True)
    def gaussian2d(x, y, sigma):
        
        result = np.exp(-(np.power(x, 2)+np.power(y, 2))/(2*np.power(sigma, 2)))
        return result

    @numba.jit(nopython = True)
    def generate(template, halfsize, sigma):

        for v_row in range(-1*halfsize, halfsize+1):
            for v_col in range(-1*halfsize, halfsize+1):
                template[v_row+halfsize, v_col+halfsize] = gaussian2d(v_row, v_col, sigma)

        element_sum = np.sum(template)
        template = template/element_sum
        
        return template

    re = generate(template, halfsize, sigma)
    return re

def drawline(image, point1, point2):
    """
    This function is used for drawing a point from point1 to point2
    
    Args:
        image: the image you want to draw lines
        point1: the origin, could be a array
        point2: the destination could be a array
    Return:
        None
    """

    if len(point1) != len(point2):
        raise Exception("The elements of point_lists should be same.")

    for i in range(len(point1)):

        cv.line(image, point1[i], point2[i], (255, 0, 0), 3)
    
    return image
    