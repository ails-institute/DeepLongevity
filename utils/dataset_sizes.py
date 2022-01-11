#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:22:27 2020

@author: ayakimovich
"""
import os
import tables
import numpy as np

def h5_read(file):
    h5file = tables.open_file(file, driver="H5FD_CORE")
    array = h5file.root.somename.read()
    #h5file.close()
    return array, h5file

load_dir=os.path.join('..tensors/jrz_data')

y_train, h5file = h5_read(os.path.join(load_dir,"labels_train.hdf5"))
h5file.close()
y_validate, h5file = h5_read(os.path.join(load_dir,"labels_validate.hdf5"))
h5file.close()
y = np.concatenate(y_train,y_validate)
print('dataset: {}, total: {}, class 0: {}, class 1: {}'.format(load_dir,len(y),len(y[y==0]),len(y[y==1])))