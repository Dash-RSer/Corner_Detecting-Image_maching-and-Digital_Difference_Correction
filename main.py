# -*- coding: utf-8 -*-
# @author: Dasheng Fan
# @email: fandasheng1999@163.com

"""
This code is the main program.
"""

import os
import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.append("./utils")
sys.path.append("./operators")
sys.path.append("./match")

from rasterio import readimage, write
from stretch import stretch
from display import display, doubledisplay, mask_process, featuredisplay
from Forstner import Forstner
from Harris import  Harris
from Canny import Canny
from ABM import ABM
from ddc.ddc import DDC
import gc

def main():

    # 1
    # imagepath = "data\ex1\DJI_0013.JPG"
    # imagepath = "data\ex1\DSC00438.JPG"
    # img = readimage(imagepath)
    # 2
    # imagepath = "F:\photography\data\ex2\DJI_0011.JPG"
    # imagepath = "F:\photography\data\ex2\DJI_0300.JPG"
    # img = readimage(imagepath)
    # 3
    imagepath1 = "data\ex3\DSC00437.JPG"
    imagepath2 = "data\ex3\DSC00438.JPG"
    img1 = readimage(imagepath1)
    img2 = readimage(imagepath2)
    ##################################
    # forstner = Forstner()    
    # mask = np.zeros((img.shape[0], img.shape[1]), np.float32)
    # mask = forstner.detect(img)
    # mask_for_show = mask_process(mask)
    # featuredisplay(mask_for_show, img)
    # ##################################
    # harris = Harris()
    # mask = harris.detect(img)
    # mask_for_show = mask_process(mask)
    # featuredisplay(mask_for_show, img)
    ##################################

    # canny = Canny()
    # mask = canny.detect(img)
    # mask_for_show = mask_process(mask)
    # doubledisplay(mask_for_show, img)
    #####################################
    # abm = ABM()
    # image = abm.match(img1, img2)
    # del abm
    # gc.collect()````````````````````````````````````
    # write(image, './abm_result')
    # display(image)
    #####################################
    ddc = DDC()
    o_origin_matrix = np.array(
        [[2506.373481024566,-7033.834063624827,-4070.333705903987,27449753615.492199000000],
        [-7014.690436111545,-2225.285921677511,-3064.489319726752,10558841260.527899000000],
        [0.048638596498,0.029800644767,-0.998371778699,-130213.693521655590]]
    )
    ddc.correct("data\ex4\DSC00437.jpg" ,"data\ex4\DEM.tif", o_origin_matrix)

if __name__ == '__main__':
    main()
