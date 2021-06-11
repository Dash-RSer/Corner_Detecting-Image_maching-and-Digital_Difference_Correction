# -*- coding: utf-8 -*-
# @author: Dasheng Fan
# @email: fandasheng1999@163.com

"""
This code is used for image matching.
....有点想用Cython写了....
"""
import numpy as np
import numba
import gc
from operators.Harris import Harris
from utils.gray_processing import gray_processing
from operators.diptools import drawline
from utils.display import display

class ABM(object):
    def __init__(self,
                window_size = 5, 
                row_offset = 1000, 
                col_offset = 0, 
                search_window_size = 800):
        """
        Args:
            lazy boy with no annotation.:)

        注意偏移量是必须的，这个数自己目视估计，不然太慢了程序。即使增加了
        numba，还是太慢。下一步准备Tensorflow写了。
        这个代码注释近乎没办法写。

        """
        self._window_size = window_size
        self._row_offset = row_offset
        self._col_offset = col_offset
        self._search_window_size = search_window_size

    def match(self, image1, image2):
        origin_img1 = image1
        origin_img2 = image2

        if len(image1.shape) != 2:
            image1 = gray_processing(image1)
        if len(image2.shape) != 2:
            image2 = gray_processing(image2)

        harris = Harris(cut_ratio=99.99)
        mask = harris.detect(image1)

        @numba.jit(nopython = True)
        def correlation_coefficient(window1, window2):
            w1_mean = np.mean(window1)
            w2_mean = np.mean(window2)

            window_size = window1.shape[0]
            covariance = np.float32(0)
            variance_origin = np.float32(0)
            variance_target = np.float32(0)
            
            for row in range(window_size):
                for col in range(window_size):
                    covariance = covariance +\
                         np.multiply(window1[row, col], window2[row, col])

                    variance_origin = variance_origin + \
                        np.power((window1[row, col] - w1_mean), 2)
                    
                    variance_target = variance_target + \
                        np.power((window2[row, col] - w2_mean), 2)

                    
                        
            if variance_target == 0 or variance_origin == 0:
                rho = 0
            else:
                rho = covariance/np.sqrt(np.multiply(variance_origin, variance_target))

            return rho
        
        @numba.jit(nopython = True)
        def _match(mask, image1, image2, window_size, row_offset, col_offset, search_window_size):
            
            height, width = mask.shape
            half_size = np.int32(np.floor(window_size/2))

            window_offset = np.int32(search_window_size/2)
            
            origin_point_list = []
            target_points_list = []

            print("start matching, please wait...")

            for row in range(half_size, height - half_size):
                if row + row_offset < height and row + row_offset > 0 and row + half_size < height and row - half_size > 0:                
                    for col in range(half_size, width - half_size):
                        if col + col_offset < width and col + col_offset > 0 and col + half_size < height and col - half_size > 0:
                            if mask[row, col] == 1:
                                origin_point_list.append((row, col))
                                origin_img_window = image1[row-half_size:row+half_size+1, col-half_size:col+half_size+1]
                                
                                rho_list = []
                                position_list = []
                                
                                
                                for row_in in range(row+row_offset-window_offset, row+row_offset+window_offset):
                                    if row_in+half_size < height and row_in-half_size > 0:
                                        for col_in in range(col+col_offset-window_offset, col+col_offset+window_offset):
                                            if col_in+half_size < width and col_in-half_size > 0:                                         
                                                target_img_window =\
                                                    image2[row_in-half_size:row_in+half_size+1, col_in-half_size:col_in+half_size+1]
                                                rho = correlation_coefficient(origin_img_window, target_img_window)
                                                rho_list.append(rho)
                                                position_list.append((row_in, col_in))
                                
                                index = np.argmax(np.array(rho_list))
                                target_points_list.append(position_list[index])

            print("finish.")

            return origin_point_list, target_points_list

        origin_point_list, target_points_list = _match(mask, image1, image2, 
                                                    self._window_size, 
                                                    self._row_offset, 
                                                    self._col_offset,
                                                    self._search_window_size)

        def print_points(origin_point_list, target_points_list):
            print("Points couples:")
            for i in range(len(origin_point_list)):
                print("{}-->{}".format(origin_point_list[i], target_points_list[i]))
            print("print over.")
        # print_points(origin_point_list, target_points_list)

        del mask, harris
        del image1, image2
        gc.collect()

        def showmatchpoints(img1, img2, origin_point_list, target_points_list):
            height, width, channels = img1.shape

            image = np.concatenate((img1, img2), axis=1)
            new_target_points_list = []
            new_origin_points_list = []
            for i in range(len(origin_point_list)):
                x_new = target_points_list[i][1] + width
                y = target_points_list[i][0]
                new_target_points_list.append((x_new, y))
                x = origin_point_list[i][1]
                y = origin_point_list[i][0]
                new_origin_points_list.append((x, y))

            image = drawline(image, new_origin_points_list, new_target_points_list)
            
            return image
        
        print("drawing...")
        image = showmatchpoints(origin_img1, origin_img2, origin_point_list, target_points_list)
        del origin_point_list, target_points_list
        print("finish.")
        del origin_img1, origin_img2
        gc.collect()
        return image




            
                                

                                
                                    
                                
                                
                                
                                

