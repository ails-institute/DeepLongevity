 # -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 14:22:59 2018

@author: Artur Yakimovich, PhD
"""
from __future__ import print_function
import numpy as np

import time
import os
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import losses
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#from keras.applications.resnet50 import ResNet50, preprocess_input#, preprocess_input
#from keras.preprocessing import image    
#import resnet

#import resnet_152
#import resnet_activity_reg
from keras.models import model_from_json
from keras import backend as K
#from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout, AveragePooling2D # Dropout, Convolution2D, Flatten, MaxPool2D
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model
from keras.applications.mobilenet import MobileNet
#from keras.applications.mobilenet_v2 import MobileNetV2
from collections import Counter
from keras import regularizers
from keras.constraints import max_norm
from sklearn.utils import class_weight



def countFiles(directory):
    fileCount = 0
    for iDir in os.listdir(directory):
        fileCount += len([name for name in os.listdir(os.path.join(directory,iDir)) if os.path.isfile(os.path.join(directory,iDir, name))])
    return fileCount

def getClassWeights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cl: float(majority/count) for cl, count in counter.items()}


    
batchSize = 100 #  4
foldAugment = 100
img_size_factor = 0.5 # 0.75
runningTime = time.strftime('%b-%d-%Y_%H-%M')
numOfClasses = 2
epochs = 200
img_width, img_height = 900, 900
channels = 3
startingLeraningRate = 1e-2 #0.00005, 0.00001
earlyStopFlag = False
modelCheckpointFlag = True
modelContinueFlag = False
reduceLRFlag = True
useRegularizer = False
fineTuneAt = 0
K.set_image_data_format('channels_last')

modelContinueFile = os.path.join("F:\\Dropbox (Personal)\\Python_codes\\v4\\May-23-2019_10-23","worm_incV3_fullres.json")
modelContinueWeigthsFile = os.path.join("F:\\Dropbox (Personal)\\Python_codes\\v4\\May-23-2019_10-23","worm_incV3_fullres.h5")

if K.image_data_format() == 'channels_first':
    input_shape = (channels, int(img_width*img_size_factor), int(img_height*img_size_factor))
else:
    input_shape = (int(img_width*img_size_factor), int(img_height*img_size_factor), channels)

modelDir = "F:\\data\\v4"
checkpointDir = modelDir+"\\"+"Checkpoints"
modelFile = modelDir+"\\"+"{}_worm_MobileNet_fullres.json".format(runningTime)
weightsFile = modelDir+"\\"+"{}_worm_MobileNet_fullres.h5".format(runningTime)

# load images

trainFolder = "F:\\data\\day3splited_subset\\train"
validateFolder = "F:\\data\\day3splited_subset\\validate"
testFolder = "F:\\data\\day3splited_subset\\test"

trainSamplesNumber = countFiles(trainFolder)
validateSamplesNumber = countFiles(validateFolder)
testSamplesNumber = countFiles(testFolder)




#model = resnet.ResnetBuilder.build_resnet_50((channels,img_width,img_height), numOfClasses) #_18 _34 _50 _101
#model = resnet_activity_reg.ResnetBuilder.build_resnet_50((channels,img_width,img_height), numOfClasses) #_18 _34 _50 _101
model = MobileNet(include_top=True, weights=None,
    #  '             input_tensor=Input(shape=input_shape))
                    alpha=0.1,
                    input_tensor=None, input_shape=input_shape, pooling='avg', classes=2)

if modelContinueFlag:
   model.load_weights(modelContinueWeigthsFile, by_name=False)
   
model.summary()
model.compile(loss='categorical_crossentropy',
              #model.compile(loss='categorical_hinge',
              optimizer=Adam(lr=startingLeraningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              #optimizer=RMSprop(lr=startingLeraningRate, rho=0.9, epsilon=None, decay=0.0),
              #optimizer=SGD(lr=startingLeraningRate, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy','categorical_accuracy']) # default lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0 or 0.00005

tensorboard = TensorBoard(log_dir=os.path.join(modelDir,"{}".format(runningTime)))
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.000001, verbose=1)
earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2000, verbose=1, mode='auto')
modelCheckpoint = ModelCheckpoint(os.path.join(checkpointDir,"weights.{epoch:02d}-{val_loss:.2f}.hdf5"), monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=50)

print("start tesorboard, cmd: tensorboard --logdir=\""+os.path.join(modelDir,"{}".format(runningTime)+"\""))


#class_weights={0:class_weights_array[0],1:class_weights_array[1]}

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=90,
    vertical_flip=True,
    #brightness_range=(0.8,0.99),
    horizontal_flip=True)

validate_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

trainGenerator = train_datagen.flow_from_directory(
    trainFolder,
    color_mode='rgb',    
    shuffle=True,
    seed=123,
    target_size=(int(img_width*img_size_factor), int(img_height*img_size_factor)),
    batch_size=batchSize,
    classes=['short','long'],
    class_mode='categorical')

validationGenerator = validate_datagen.flow_from_directory(
    validateFolder,
    color_mode='rgb',
    shuffle=True,
    seed=123,
    target_size=(int(img_width*img_size_factor), int(img_height*img_size_factor)),
    batch_size=batchSize,
    classes=['short','long'],
    class_mode='categorical')

testGenerator = test_datagen.flow_from_directory(
    testFolder,
    color_mode='rgb',
    shuffle=True,    
    seed=123,
    target_size=(int(img_width*img_size_factor), int(img_height*img_size_factor)),
    batch_size=batchSize,
    classes=['short','long'],
    class_mode='categorical')

callbacksList = [tensorboard]

if earlyStopFlag:
    callbacksList.append(earlyStop)
if reduceLRFlag:    
    callbacksList.append(reduce_lr)
if modelCheckpointFlag:
    callbacksList.append(modelCheckpoint)


    
history = model.fit_generator(
                trainGenerator,
                steps_per_epoch=trainSamplesNumber // batchSize * foldAugment,
                epochs=epochs,
                verbose=1,
                callbacks=callbacksList,
                validation_data=validationGenerator,
                class_weight=getClassWeights(trainGenerator.classes),
                shuffle=True,
                validation_steps=validateSamplesNumber // batchSize)



score = model.evaluate_generator(testGenerator, testSamplesNumber)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# serialize model to JSON
model_json = model.to_json()
with open(modelFile, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(weightsFile)
print("Saved model to disk")