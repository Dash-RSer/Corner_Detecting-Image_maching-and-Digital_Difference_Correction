# -*- coding: utf-8 -*-
# @author: Dasheng Fan
# @email: fandasheng1999@163.com

"""
This code is for the image displaying.
"""

import numpy as np
import numba as nb
from matplotlib import pyplot as plt
from utils.stretch import stretch

def display(image):
    """
    This function is used for show a image
    
    Args:
        image: The image your want to display
    """
    
    image = image.astype(np.float16)
    image = stretch(image, max_value=1)

    if len(image.shape) == 3:
        plt.imshow(image)
    
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')

    plt.show()

def doubledisplay(image1, image2):
    """
    This function is used for show two images for comparation.
    """
    image1 = stretch(image1.astype(np.float32), 1)
    image2 = stretch(image2.astype(np.float32), 1)

    plt.figure()
    i = 1
    for image in [image1, image2]:
        plt.subplot(1, 2, i)
        if len(image.shape) == 3:
            plt.imshow(image)
        elif len(image.shape) == 2:
            plt.imshow(image, cmap = 'gray')
        i+=1
    plt.show()


def featuredisplay(mask, image):
    height, width = mask.shape
    mask = mask_process(mask)
    @nb.jit(nopython = True)
    def proc(image, mask):
        height, width = mask.shape
        for row in range(height):
            for col in range(width):
                if mask[row, col] == 1:
                    image[row, col, 0] = 255
                    image[row, col, 1] = 0
                    image[row, col, 2] = 0
        return image

    image = proc(image, mask)
    display(image)
    


def mask_process(mask):
    """
    This function is used to change the flag points
    of the mask to a '+' smybol for better perception.

    Args:
        mask:a binary numpy array as a mask.
    
    Return:
        a processed mask with '+' in every '1' points.
    """
    height = mask.shape[0]
    width = mask.shape[1]

    extend_rol_and_col = 5
    new_mask = np.zeros((height, width), dtype = np.float32)
    
    @nb.jit(nopython = True)
    def change(mask, new_mask, height, width):
        
        for row in range(extend_rol_and_col, height-extend_rol_and_col):
            for col in range(extend_rol_and_col, width-extend_rol_and_col):
                if mask[row, col] == 1:
                    for i in range(extend_rol_and_col*-1, extend_rol_and_col+1):
                        new_mask[row+i, col] = 1
                        new_mask[row, col+i] = 1

    change(mask, new_mask, height, width)

    return new_mask


    


