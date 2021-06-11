# -*- coding: utf-8 -*-
# @author: Dasheng Fan
# @email: fandasheng1999@163.com

"""
Canny operator.
"""
import numpy as np
import numba
from operators.diptools import generate_Guassian_template, convolution
from operators.gradient import RobertsGradient,Gradient
from utils.gray_processing import gray_processing
from utils.display import display, doubledisplay


class Canny(object):
    """
    Canny operator is used for line detecting.
    """

    def __init__(self, 
            Guassian_kernel_size = 3, 
            Guassian_sigma = 1, 
            high_threshhold_ratio = 0.15,
            low_threshhold_ratio = 0.05):

        self._G_kernel_size = Guassian_kernel_size
        self._G_sigma = Guassian_sigma
        self._h_thresh_r = high_threshhold_ratio
        self._l_thresh_r = low_threshhold_ratio

    def detect(self, image):

        if len(image.shape) != 2:
            image = gray_processing(image)

        height, width = image.shape

        g_template = generate_Guassian_template()
        image = convolution(image, g_template)

        gradient = Gradient()
        gx, gy = gradient.calculate(image)

        @numba.jit(nopython = True)
        def nonmaxsuppress(gx,gy):
            """
            Reference:
            https://blog.csdn.net/kezunhai/article/details/11620357
            """
            
            g = np.sqrt(np.add(np.power(gx, 2), np.power(gy, 2)))
            angle = np.arctan(gy/gx)
            
            height, width = g.shape
            flags = np.zeros((height, width), dtype = np.float32)

            for row in range(1, height-1):
                for col in range(1, width - 1):

                    local_g = g[row, col]

                    if np.abs(gy[row, col])>np.abs(gx[row, col]):
                        
                        if gy[row, col] == 0:
                            weight = 1
                        else:
                            weight = np.abs(gx[row,col]/gy[row, col])
                        g2 = g[row-1, col]
                        g4 = g[row+1, col]
                        
                        if np.multiply(gx[row,col], gy[row, col])>0:
                            g1 = g[row-1, col-1]
                            g3 = g[row+1, col+1]
                        else:
                            g1 = g[row-1, col+1]
                            g3 = g[row+1, col-1]
                    else:
                        if gx[row, col] == 0:
                            weight = 1
                        else:
                            weight = np.abs(gy[row, col]/gx[row, col])
                        g2 = g[row, col-1]
                        g4 = g[row, col+1]

                        if np.multiply(gx[row, col],gy[row,col])>0:
                            g1 = g[row-1, col-1]
                            g3 = g[row+1, col+1]
                        else:
                            g1 = g[row+1, col-1]
                            g3 = g[row-1, col+1]

                    inter_g1 = weight*g1 + (1-weight)*g2
                    inter_g2 = weight*g3 + (1-weight)*g4 
                    local_g = g[row, col]

                    if local_g >=inter_g1 and local_g>=inter_g2:
                        flags[row, col] = 1
            
            return flags

        flags1 = nonmaxsuppress(gx, gy)
        
        @numba.jit(nopython = True)
        def _double_threshhold_suppress(
                                high_threshhold_ratio, 
                                low_threshhold_ratio, 
                                gx, gy):
            """
            The theory can be found here:
            https://www.cnblogs.com/techyan1990/p/7291771.html
            Only the theory is refered, the codes are different.
            """

            g = np.sqrt(np.add(np.power(gx, 2), np.power(gy, 2)))
            height, width = g.shape
            max_g = np.max(g)
            high_thresh = max_g * high_threshhold_ratio
            low_thresh = max_g * low_threshhold_ratio

            flags = np.zeros((height, width), dtype = np.float32)
            
            for row in range(1, height-1):
                for col in range(1, width - 1):
                    
                    if g[row, col] >= high_thresh:
                        flags[row, col] = 1
                    elif g[row, col] > low_thresh and g[row, col] < high_thresh:
                        # 不洋嚯了，写汉语
                        # 这里检查一下八邻域， 如果有强边缘就认为弱边缘是边缘点
                        for var_y in range(-1, 2):
                            for var_x in range(-1, 2):
                                if g[row+var_y, row+var_x] > high_thresh:
                                    flags[row, col] = 1
                                    break
                    else:
                        flags[row, col] = 0
            
            return flags

        flags2 = _double_threshhold_suppress(self._h_thresh_r, self._l_thresh_r, gx, gy)

        flags = np.multiply(flags1, flags2)

        return flags
        







                        











        