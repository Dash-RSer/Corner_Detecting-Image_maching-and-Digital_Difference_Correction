# -*- coding: utf-8 -*-
# @author: Dasheng Fan
# @email: fandasheng1999@163.com

"""
Strech a image into a range (from 0 to some number).
"""

import numpy as np

def stretch(image, max_value = 1):
    
    if len(image.shape) == 2:
        image = image.astype(np.float)

        max_val = np.max(image)
        min_val = np.min(image)

        diff = max_val - min_val

        if diff == 0:
            raise Exception("The range of the image is 0.")

        image = image * max_value/diff
    
    elif len(image.shape) == 3:
        image = image.astype(np.float)

        for band in range(len(image.shape)):
            max_val = np.max(image[:, :, band])
            min_val = np.min(image[:, :, band])
            diff = max_val - min_val

            if diff == 0:
                raise Exception("The range of the image is 0.")

            image[:, :, band] = image[:, :, band]*max_value/diff
    
    else:
        raise Exception("The dimention of the image is wrong!")
    
    return image
