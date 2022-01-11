# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 09:33:39 2018

@author: Artur Yakimovich, PhD
"""

# from __future__ import print_function

import os
import pandas as pd
from shutil import copyfile

def checkDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
def dataSplitter(validateFactor, testFactor, dataSamples, table, seed=123):
    #train = table.iloc[:int(validateFactor*dataSamples),:]
    #validate = table.iloc[int(validateFactor*dataSamples):int(testFactor*dataSamples),:]
    #test = table.iloc[int(testFactor*dataSamples):,:]
    train = table.sample(n=int(validateFactor*dataSamples), random_state=seed)
    test = table.drop(train.index).sample(n=int((1-testFactor)*dataSamples), random_state=seed)
    validate = table.drop(train.index).drop(test.index)    
    
    return train, validate, test

def dataSplitterAbs(valSize, testSize, dataSamples, table, seed=123):
    #train = table.iloc[:int(dataSamples-valSize-testSize),:]
    #validate = table.iloc[int(dataSamples-valSize-testSize):int(dataSamples-testSize),:]
    #test = table.iloc[int(dataSamples-testSize):,:]
    train = table.sample(n=dataSamples-valSize-testSize, random_state=seed)
    test = table.drop(train.index).sample(n=testSize, random_state=seed)
    validate = table.drop(train.index).drop(test.index) 
    
    return train, validate, test

def makeClassSubdir(rootDir, classes):
    for classDir in classes:
        dirToMake = os.path.join(rootDir,classDir)
        if not os.path.exists(dirToMake):
            os.makedirs(dirToMake)

def sortImages(df, rootDir, fraction):
    for i,iPath in enumerate(range(len(df['3_max_bulk_movement_cat']))):
        print(i)
        #src = os.path.join(df['path'].iloc[iPath],df['file'].iloc[iPath])
        src = os.path.join(df['pathNewData'].iloc[iPath],df['fileNewData'].iloc[iPath].replace('_small',''))
        dst = os.path.join(rootDir, fraction, df['3_max_bulk_movement_cat'].iloc[iPath], '{}_'.format(i)+df['fileNewData'].iloc[iPath].replace(" ", "__"))
        print("copy:\n"+src+"\n to:\n "+dst)
        copyfile(src, dst)

splitFractionTrainValidate = 0.8
splitFractionValidateTest = 0.9
modelDir = ""
dataDir = "F:\\data\\day3splited_mov"
checkDir(dataDir)
trainDir = os.path.join(dataDir,"train")
checkDir(trainDir)
validateDir = os.path.join(dataDir,"validate")
checkDir(validateDir)
testDir = os.path.join(dataDir,"test")
checkDir(testDir)



# load data table
newLabelsFile = os.path.join(modelDir,"3days_newlables.csv")
dataFile = os.path.join(modelDir,"data_mid_point_3days_new_data_lables_subset.csv")
dataTable = pd.read_table(dataFile, delimiter=",")
newLabelsTable = pd.read_table(newLabelsFile, delimiter=",")
newLabelsTable['id_path'] = "F:\\data\\crops1\\"+newLabelsTable['1_id']
dataTable = pd.merge(dataTable, newLabelsTable, left_on='pathNewData', right_on='id_path', how='left')

dataTable = dataTable.dropna()

classNames = dataTable['3_max_bulk_movement_cat'].unique().tolist()

makeClassSubdir(trainDir,classNames)
makeClassSubdir(validateDir,classNames)
makeClassSubdir(testDir,classNames)
# Split data table
longLength = len(dataTable.loc[dataTable['3_max_bulk_movement_cat'] == 'high', '3_max_bulk_movement_cat'])
#mediumLength = len(dataTable.loc[dataTable['lifeSpanDuration'] == 'medium', 'lifeSpanHours'])
shortLength = len(dataTable.loc[dataTable['3_max_bulk_movement_cat'] == 'low', '3_max_bulk_movement_cat'])

longData = dataTable.loc[dataTable['3_max_bulk_movement_cat'] == 'high', ('3_max_bulk_movement_cat', 'pathNewData', 'fileNewData') ]
#mediumData = dataTable.loc[dataTable['lifeSpanDuration'] == 'medium', ('lifeSpanDuration', 'path', 'file') ]
shortData = dataTable.loc[dataTable['3_max_bulk_movement_cat'] == 'low', ('3_max_bulk_movement_cat', 'pathNewData', 'fileNewData') ]

if(longLength <= shortLength):
    longTrainData, longValidateData, longTestData = dataSplitter(splitFractionTrainValidate, splitFractionValidateTest, longLength, longData)
    shortTrainData, shortValidateData, shortTestData = dataSplitterAbs(len(longValidateData), len(longTestData), shortLength, shortData)

elif(longLength > shortLength):
    shortTrainData, shortValidateData, shortTestData = dataSplitter(splitFractionTrainValidate, splitFractionValidateTest, shortLength, shortData)
    longTrainData, longValidateData, longTestData = dataSplitterAbs(len(shortValidateData), len(shortTestData), longLength, longData)

sortImages(longTrainData, dataDir, 'train')
sortImages(longValidateData, dataDir, 'validate')
sortImages(longTestData, dataDir, 'test')

#mediumTrainData, mediumValidateData, mediumTestData = dataSplitter(splitFractionTrainValidate, splitFractionValidateTest, mediumLength, mediumData)
#sortImages(mediumTrainData, dataDir, 'train')
#sortImages(mediumValidateData, dataDir, 'validate')
#sortImages(mediumTestData, dataDir, 'test')

shortTrainData, shortValidateData, shortTestData = dataSplitter(splitFractionTrainValidate, splitFractionValidateTest, shortLength, shortData)
sortImages(shortTrainData, dataDir, 'train')
sortImages(shortValidateData, dataDir, 'validate')
sortImages(shortTestData, dataDir, 'test')


#trainData = pd.concat([longTrainData, mediumTrainData, shortTrainData], ignore_index=True, join='outer').sample(frac=1).reset_index(drop=True)
#trainData = pd.concat([trainData, pd.get_dummies(trainData['lifeSpanDuration'])], axis=1)
#validateData = pd.append([longValidateData, mediumValidateData, shortValidateData]).sample(frac=1).reset_index(drop=True)
#validateData = pd.concat([validateData, pd.get_dummies(validateData['lifeSpanDuration'])], axis=1)
#testData = pd.append([longTestData, mediumTestData, shortTestData]).sample(frac=1).reset_index(drop=True)
#testData = pd.concat([testData, pd.get_dummies(testData['lifeSpanDuration'])], axis=1)