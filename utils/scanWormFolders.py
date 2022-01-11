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
dfFile = "R:\\Dropbox\\Artur-Evgenij\\V4\\data_lables.csv"
dfMidPointFile = "R:\\Dropbox\\Artur-Evgenij\\V4\\data_mid_point_1days_lables.csv"
dataFolder = "R:\\Data\\180321_DeepLongevity\\EvgenyMasks"
data = pd.DataFrame()

for root, dirs, files in os.walk(dataFolder):
    for file in files:
        if file.endswith("_small.png"):
             #print(os.path.join(root, file))
             data = data.append({'path': root, 'file': file}, ignore_index=True)

data['date'] = data['file'].apply(lambda x:
    pd.to_datetime(re.sub(r'\scrop_small.png','', x),
                   format='%Y-%m-%dt%H%M'))
wormID = 0
data['lifeSpanHours'] = 0
data['lifeSpanDays'] = 0
data['elapsedHours'] = 0
data['elapsedDays'] = 0
data['wormID'] = ''

for iWorm in data['path'].unique():
    data.loc[data['path'] == iWorm, 'lifeSpanDays'] =  (data.loc[data['path'] == iWorm, 'date'].max()-data.loc[data['path'] == iWorm, 'date'].min()).days
    data.loc[data['path'] == iWorm, 'lifeSpanHours'] =  (data.loc[data['path'] == iWorm, 'date'].max()-data.loc[data['path'] == iWorm, 'date'].min()).seconds/3600 + data.loc[data['path'] == iWorm, 'lifeSpanDays'] * 24
    data.loc[data['path'] == iWorm, 'elapsedDays'] =  (data.loc[data['path'] == iWorm, 'date'] - data.loc[data['path'] == iWorm, 'date'].min())/(86400*1e9)
    data.loc[data['path'] == iWorm, 'elapsedHours'] =  (data.loc[data['path'] == iWorm, 'date'] - data.loc[data['path'] == iWorm, 'date'].min())/(3600*1e9)
    data.loc[data['path'] == iWorm, 'wormID'] = '%d' % wormID
    wormID += 1

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