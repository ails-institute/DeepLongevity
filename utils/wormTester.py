# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 14:22:59 2018

@author: Artur Yakimovich, PhD
"""

from __future__ import print_function
import time
import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import model_from_json
from collections import Counter
from keras.optimizers import Adam, SGD

def countFiles(directory):
    fileCount = 0
    for iDir in os.listdir(directory):
        fileCount += len([name for name in os.listdir(os.path.join(directory,iDir)) if os.path.isfile(os.path.join(directory,iDir, name))])
    return fileCount

def getClassWeights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cl: float(majority/count) for cl, count in counter.items()}
    
batchSize = 3
runningTime = time.strftime('%b-%d-%Y_%H-%M')
numOfClasses = 3
epochs = 3000
img_width, img_height = 800, 800
channels = 3
startingLeraningRate = 0.00001 #0.00005
modelContinueFile = os.path.join("R:\\Data\\180321_DeepLongevity\\Models\\v4","Aug-18-2018_20-41_worm_IncepV3_0_77.json")
modelContinueWeigthsFile = os.path.join("R:\\Data\\180321_DeepLongevity\\Models\\v4","Aug-18-2018_20-41_worm_IncepV3_0_77.h5")


modelDir = "R:\\Data\\180321_DeepLongevity\\Models"

testFolder = "R:\\Data\\180321_DeepLongevity\\day3_data_fullres\\test"

testSamplesNumber = countFiles(testFolder)




with open(modelContinueFile, 'r', encoding='utf-8') as f:
   model = model_from_json(f.read())
model.load_weights(modelContinueWeigthsFile, by_name=False)
   #startingLeraningRate = 0.00001
model.summary()
model.compile(loss='categorical_crossentropy',
              #model.compile(loss='categorical_hinge',
              optimizer=Adam(lr=startingLeraningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              #optimizer=SGD(lr=startingLeraningRate, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy','categorical_accuracy'])

test_datagen = ImageDataGenerator(rescale=1. / 255)


testGenerator = test_datagen.flow_from_directory(
    testFolder,
    shuffle=False,    
    seed=0,
    target_size=(img_width, img_height),
    batch_size=batchSize,
    classes=['short','long'],
    class_mode='categorical')


score = model.evaluate_generator(testGenerator, testSamplesNumber)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# serialize model to JSON
#model_json = model.to_json()
#with open(modelFile, "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights(weightsFile)
#print("Saved model to disk")