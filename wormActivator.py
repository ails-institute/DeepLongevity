# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:18:05 2018

@author: Arthur
"""

import numpy as np  
import matplotlib.pyplot as plt
import cv2       
import os
from keras.models import model_from_json
from vis.utils  import utils
from vis.visualization import visualize_cam,visualize_saliency,get_num_filters
import re



def plotNSave(im,file,grads):
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(im, alpha=1)
    ax[1].imshow(grads, cmap='jet')
    ax[2].imshow(grads, cmap='jet')
    ax[2].imshow(im, alpha=0.5)
    f.savefig(file, bbox_inches='tight')

def listImages(folder, endPattern):
    data = list()
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(endPattern):
                 data.append(file)
    return data

modelFolder = "/Users/ayakimovich/Dropbox (Personal)/Artur-Evgenij/v4"
imageFolder = "/Users/ayakimovich/Dropbox (Personal)/Artur-Evgenij/v4/day3_data_fullres/validate/long"
modelFile = os.path.join(modelFolder,"Aug-18-2018_20-41_worm_IncepV3_0_82.json")
modelWeigthsFile = os.path.join(modelFolder,"Aug-18-2018_20-41_worm_IncepV3_0_82.h5")
with open(modelFile, 'r', encoding='utf-8') as f:
    model = model_from_json(f.read())
model.load_weights(modelWeigthsFile, by_name=False)
model.summary()

images = listImages(imageFolder,".png")
             
layer_idx = utils.find_layer_idx(model,'avg_pool')
plt.rcParams['figure.figsize']=(18,6)

for image in images:
    imgPath = os.path.join(imageFolder,image)
    im = cv2.resize(cv2.imread(imgPath), (800, 800))
    im = np.round(im.astype(float)*2.6).astype(int)
    #plt.imshow(im)
    plt.figure()
    modifier = 'guided'
    #modifier = 'relu'
    #grads = visualize_saliency(model, layer_idx, filter_indices=0, seed_input=im, backprop_modifier=modifier)
    grads = visualize_cam(model, layer_idx, filter_indices=None, seed_input=im, backprop_modifier=modifier)
    file = re.sub(r"png", "pdf",imgPath)
    plotNSave(im,file,grads)