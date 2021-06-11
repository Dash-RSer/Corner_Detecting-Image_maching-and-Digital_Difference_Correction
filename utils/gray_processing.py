
# -*- coding: utf-8 -*-
# @author: Dasheng Fan
# @email: fandasheng1999@163.com

"""
Gray processing.
"""

import numpy as np
from matplotlib import pyplot as plt
import sys
import os

def gray_processing(image):
    """
    For gray processing.
    
    Args:
        image: A 3 channels image.
    
    Reture:
        A gray image.

    Reference:
    "https://baike.baidu.com/item/%E7%81%B0%E5%BA%A6%E5%8C%96/3206969?fr=aladdin"
    """

    if len(image.shape) != 3:
        raise Exception("The channel is wrong.")

    u = np.power(image[:, :, 0], 2.2) + np.power(1.5*image[:, :, 1], 2.2) + \
        np.power(0.6*image[:, :, 2], 2.2)

    d = 1 + np.power(1.5, 2.2) + np.power(0.6, 2.2)

    gray = np.power(u/d, 1/2.2)

    return gray

if __name__ == '__main__':
    pass




