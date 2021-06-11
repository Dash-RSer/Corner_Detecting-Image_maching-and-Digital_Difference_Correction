# -*- coding: utf-8 -*-
# @author: Dasheng Fan
# @email: fandasheng1999@163.com

"""
For digital difference correction
The code is very stupid, but it works well.

为了清晰的理顺思路，本代码使用汉语注释
和一般的数字微分纠正步骤不同，使用了特殊的方法
投影矩阵参数位于'../data/ex4/DSC00437.txt'中
DEM和原始图像都在ex4文件夹中
"""

import numpy as np
import numba
from utils import rasterio, gray_processing

class DDC(object):
    def __init__(self):
        self._save_path = ""

    def correct(self, image_path, dem_path, o_projection_matrix, save_path = "result"):
        if save_path == "":
            self._save_path = "result"

        image = rasterio.readimage(image_path, False)
        if len(image.shape) == 3:
            image = gray_processing.gray_processing(image)
        img_height, img_width = image.shape

        dem, projection, geo_information = rasterio.readimage(dem_path, return_geoinfo=True)
        dem = np.reshape(dem, (dem.shape[0], dem.shape[1]))
        dem_height, dem_width= dem.shape

        origin_x = geo_information[0]
        origin_y = geo_information[3]
        resolution = geo_information[1]

        dem_x_max = origin_x + resolution*dem_width
        dem_y_max = origin_y - resolution*dem_height
        
        H = np.mean(dem)
        o_projection_matrix = np.float32(o_projection_matrix)
        add_row = np.array([[0,0,1,-H]], dtype = np.float32)

        projection_matrix = np.concatenate((o_projection_matrix, add_row), axis = 0)
        projection_matrix_inverse = np.linalg.inv(projection_matrix)

        corner_origin_cor_left_top = \
            np.array([[0], [0], [1], [0]], dtype = np.float32)
        corner_origin_cor_right_top = \
            np.array([[img_width-1], [0], [1], [0]], dtype = np.float32)
        corner_origin_cor_left_bottom = \
            np.array([[0], [img_height-1], [1], [0]], dtype = np.float32)
        corner_origin_cor_right_bottom = \
            np.array([[img_width-1], [img_height-1], [1], [0]], dtype = np.float32)

        # 旋转矩阵*角点向量
        corner_geo_cor_left_top = \
            np.matmul(projection_matrix_inverse, corner_origin_cor_left_top)
        corner_geo_cor_right_top = \
            np.matmul(projection_matrix_inverse, corner_origin_cor_right_top)
        corner_geo_cor_left_bottom = \
            np.matmul(projection_matrix_inverse, corner_origin_cor_left_bottom)
        corner_geo_cor_right_bottom = \
            np.matmul(projection_matrix_inverse, corner_origin_cor_right_bottom)

        # 四个大地坐标，进行归一化处理 第四个元素应为1
        corner_geo_cor_left_top = \
            corner_geo_cor_left_top/corner_geo_cor_left_top[3,0]
        corner_geo_cor_right_top = \
            corner_geo_cor_right_top/corner_geo_cor_right_top[3,0]
        corner_geo_cor_left_bottom = \
            corner_geo_cor_left_bottom/corner_geo_cor_left_bottom[3,0]
        corner_geo_cor_right_bottom = \
            corner_geo_cor_right_bottom/corner_geo_cor_right_bottom[3,0]

        x_cor = np.array([corner_geo_cor_left_top[0,0],
                         corner_geo_cor_right_top[0,0],
                         corner_geo_cor_left_bottom[0,0], 
                         corner_geo_cor_right_bottom[0,0]], dtype = np.float32)
        
        y_cor = np.array([corner_geo_cor_left_top[1,0],
                         corner_geo_cor_right_top[1,0],
                         corner_geo_cor_left_bottom[1,0], 
                         corner_geo_cor_right_bottom[1,0]], dtype = np.float32)

        x_min = np.min(x_cor)
        x_max = np.max(x_cor)
        y_min = np.min(y_cor)
        y_max = np.max(y_cor)

        new_width = np.int16(np.floor((x_max - x_min)/resolution))
        new_height = np.int16(np.floor((y_max - y_min)/resolution))

        new_img_matrix = np.zeros((new_height, new_width, 3), dtype = np.float32)
        
        # 重新读取三波段影像
        image = rasterio.readimage(image_path)

        # @numba.jit(nopython = True)
        def get_value(dem, 
                    new_matrix, 
                    origin_x, 
                    origin_y, 
                    dem_x_max, 
                    dem_y_max,
                    x_min,
                    x_max,
                    y_min, 
                    y_max,
                    resolution, 
                    H_mean,
                    o_projection_matrix,
                    image):
            height = image.shape[0]
            width = image.shape[1]
            new_height = new_matrix.shape[0]
            new_width = new_matrix.shape[1]
            dem_height, dem_width = dem.shape
            
            for y in range(new_height):
                for x in range(new_width):
                    x_cor = x_min + x*resolution
                    y_cor = y_max - y*resolution

                    H = np.float32(0)

                    if x_cor < origin_x or x_cor > dem_x_max or\
                         y_cor < origin_y or y_cor > dem_y_max:
                        H = H_mean
                    else:
                        # 求取当前点在DEM上的行列号

                        i = np.int16(np.floor(dem_width*(x_cor - origin_x)/(dem_x_max-origin_x)))
                        j = np.int16(np.floor(dem_height*(y_cor - origin_y)/(dem_y_max-origin_y)))

                        # 先y后x
                        H = dem[j, i]

                    geo_cor_vector =\
                        np.array([[x_cor], [y_cor], [H], [1]], dtype = np.float32)


                    origin_cor_vector =\
                        np.matmul(o_projection_matrix, geo_cor_vector)
                    
                    # normalization
                    origin_cor_vector = origin_cor_vector/origin_cor_vector[2,0]

                    origin_i = origin_cor_vector[0,0]
                    origin_j = origin_cor_vector[1,0]


                    if origin_i < 0 or origin_i > width-1 or\
                        origin_j < 0 or origin_j > height-1:
                        new_matrix[y, x, :] = 0
                    else:
                        o_x_min = np.int16(np.floor(origin_i))
                        o_x_max = np.int16(np.ceil(origin_i))
                        o_y_min = np.int16(np.floor(origin_j))
                        o_y_max = np.int16(np.ceil(origin_j))

                        if origin_i - o_x_min < 0.5 and origin_j - o_y_min < 0.5:
                            new_matrix[y, x, :] = image[o_y_min, o_x_min, :]
                        elif origin_i -o_x_min >= 0.5 and origin_j - o_y_min <0.5:
                            new_matrix[y, x, :] = image[o_y_max, o_x_min, :]
                        elif origin_i - o_x_min < 0.5 and origin_j - o_y_min >= 0.5:
                            new_matrix[y, x, :] = image[o_y_min, o_x_max, :]
                        elif origin_i - o_x_min >= 0.5 and origin_j - o_y_min >= 0.5:
                            new_matrix[y, x, :] = image[o_y_max, o_x_max, :]

            return new_matrix
        new_matrix = get_value(dem, new_img_matrix, origin_x, origin_y,
                            dem_x_max, dem_y_max, x_min, x_max, y_min, y_max, 
                            resolution, H, o_projection_matrix, image)

        new_geotransform = (x_min, resolution, 0, y_max, 0, resolution)
        rasterio.write_with_geoinfo(new_matrix, "result", new_geotransform, projection)


                    












