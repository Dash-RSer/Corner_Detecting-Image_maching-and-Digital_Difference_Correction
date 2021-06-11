
# -*- coding: utf-8 -*-
# @author: Dasheng Fan
# @email: fandasheng1999@163.com

"""
This code is used for image opening and writing.
And GDAL and numpy are used.
"""

import numpy as np
from osgeo import gdal

def readimage(path, return_geoinfo = False):
    """Read a image into a numpy array.
    
    Args:
        path: The path of image.
    
    Return:
        A numpy array of the image.
    """



    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    if dataset == None:
        raise Exception("File name error.")
    else:
        datatype = np.float
        height = dataset.RasterYSize
        width = dataset.RasterXSize
        channels = dataset.RasterCount
        # The projection and geo-transform are not considerd

        image = np.zeros((height, width, channels), dtype = datatype)
        print("read image...")
        for channel in range(channels):
            band = dataset.GetRasterBand(channel + 1)
            image[:, :, channel] = band.ReadAsArray()
        
        print("finish.")
        projection = dataset.GetProjection()
        geotransform = dataset.GetGeoTransform()
    if return_geoinfo == False:
        return image
    else:
        return image, projection, geotransform

def write(image, path, datatype = 'GTiff'):
    
    dtype = gdal.GDT_Float32

    height = image.shape[0]
    width = image.shape[1]
    if len(image.shape) == 3:
        channels = image.shape[2]
    else:
        channels = 1

    
    driver = gdal.GetDriverByName(datatype)

    ds_to_save = driver.Create(path, width, height, channels, dtype)

    print("saving image......")
    for band in range(channels):
        ds_to_save.GetRasterBand(band + 1).WriteArray(image[:, :, band])
        ds_to_save.FlushCache()

    print("saved.")
    del image
    del ds_to_save

def write_with_geoinfo(image, path, geotransform, projection ,datatype = 'GTiff'):
    
    dtype = gdal.GDT_Float32

    height = image.shape[0]
    width = image.shape[1]
    if len(image.shape) == 3:
        channels = image.shape[2]
    else:
        channels = 1

    
    driver = gdal.GetDriverByName(datatype)

    ds_to_save = driver.Create(path, width, height, channels, dtype)
    ds_to_save.SetGeoTransform(geotransform)
    ds_to_save.SetProjection(projection)
    # This is no projection and geo-transoform, we don't set them!

    print("saving image......")
    for band in range(channels):
        ds_to_save.GetRasterBand(band + 1).WriteArray(image[:, :, band])
        ds_to_save.FlushCache()

    print("saved.")
    del image
    del ds_to_save