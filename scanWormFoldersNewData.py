# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 12:27:31 2018

@author: Artur Yakimovich, PhD
"""

import os
import re
import pandas as pd
import ggplot as gg
#import ggplot

#dfFile = "R:\\Dropbox\\Artur-Evgenij\data_lables.csv"
#dfMidPointFile = "R:\\Dropbox\\Artur-Evgenij\\data_mid_point_lables.csv"
oldDfFile = "data_lables.csv"
dfFile = "data_new_data_lables_evgenij.csv"
dfMidPointFile = "data_mid_point_1days_new_data_lables_subset.csv"
dataFolder = "F:\\data\\crops1"
oldData = pd.read_table(oldDfFile, delimiter=",")
data = pd.DataFrame()
oldDataZero = oldData.loc[(oldData['elapsedDays'] == 0) & (oldData['elapsedHours'] == 0)]

print('processing folders... (may take a while)')
for root, dirs, files in os.walk(dataFolder):
    for file in files:
        if file.endswith("_.png"):
             #print(os.path.join(root, file))
             data = data.append({'pathNewData': root, 'fileNewData': file}, ignore_index=True)

data['dateNewData'] = data['fileNewData'].apply(lambda x:
    pd.to_datetime(re.sub(r'_bf.*','', x),
                   format='%Y-%m-%dt%H%M'))
print('all folders processed')
wormID = 0

data['NewDataElapsedHours'] = 0
data['NewDataElapsedDays'] = 0
data['NewDataWormID'] = ''


for iWorm in data['pathNewData'].unique():
    data.loc[data['pathNewData'] == iWorm, 'NewDataElapsedDays'] =  (data.loc[data['pathNewData'] == iWorm, 'dateNewData'] - data.loc[data['pathNewData'] == iWorm, 'dateNewData'].min())/(86400*1e9)
    data.loc[data['pathNewData'] == iWorm, 'NewDataElapsedHours'] =  (data.loc[data['pathNewData'] == iWorm, 'dateNewData'] - data.loc[data['pathNewData'] == iWorm, 'dateNewData'].min())/(3600*1e9)
    data.loc[data['pathNewData'] == iWorm, 'NewDataWormID'] = '%d' % wormID
    wormID += 1
data['NewDataWormIDNum'] = pd.to_numeric(data['NewDataWormID'], downcast='integer')
mergeData = pd.merge(data, oldDataZero, how='left', left_on='NewDataWormIDNum', right_on='wormID')

mergeData['date'] = pd.to_datetime(mergeData['date'], format='%Y-%m-%d %H:%M')
    
for iWorm in mergeData['wormID'].unique():
    mergeData.loc[mergeData['wormID'] == iWorm, 'elapsedDays'] =  (mergeData.loc[mergeData['wormID'] == iWorm, 'dateNewData'] - mergeData.loc[mergeData['wormID'] == iWorm, 'date'].min())/(86400*1e9)
    mergeData.loc[mergeData['wormID'] == iWorm, 'elapsedHours'] =  (mergeData.loc[mergeData['wormID'] == iWorm, 'dateNewData'] - mergeData.loc[mergeData['wormID'] == iWorm, 'date'].min())/(3600*1e9)

mergeData.to_csv(dfFile, sep=',', encoding='utf-8', header=True, index=False)

dataMidPoint = mergeData.loc[mergeData['NewDataElapsedDays'] == 1 , :]
dataMidPoint.to_csv(dfMidPointFile, sep=',', encoding='utf-8', header=True, index=False)
'''
data['lifeSpanDuration'] = pd.cut(data.lifeSpanDays, 2, labels=["short", "long"])
data.to_csv(dfFile, sep=',', encoding='utf-8', header=True, index=False)

data.lifeSpanDays.mean()
data.lifeSpanDays.max()
data.lifeSpanDays.min()

#dataMidPoint = data.loc[data['elapsedDays'] ==4 , :]
dataMidPoint = data.loc[data['elapsedDays'] ==1 , :]
dataMidPoint.to_csv(dfMidPointFile, sep=',', encoding='utf-8', header=True, index=False)

p = gg.ggplot(gg.aes(x='lifeSpanDuration'), data=dataMidPoint)
p + gg.geom_bar()
'''