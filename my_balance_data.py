# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 22:58:19 2018

@author: Kedrowsky
"""

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

train_data = np.load('training_data_npy.npy')
print(len(train_data))
df=pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

lefts=[]
rights=[]
forwards=[]
lefts_acc=[]
rights_acc=[]
lefts_brake=[]
rights_brake=[]
brake=[]

shuffle(train_data)

for data in train_data:
    img =data[0]
    choice=data[1]

    if choice==[1,0,0,0]:
        lefts.append([img, choice])
        
    if choice==[0,1,0,0]:
        forwards.append([img, choice])
        
    if choice==[0,0,1,0]:
        rights.append([img, choice])
        
    if choice==[1,1,0,0]:
        lefts_acc.append([img, choice])
        
    if choice==[0,1,1,0]:
        rights_acc.append([img, choice])
        
    if choice==[1,0,0,1]:
        lefts_brake.append([img, choice])
        
    if choice==[0,0,1,1]:
        rights_brake.append([img, choice])

    if choice==[0,0,0,1]:
        brake.append([img, choice])
        
forwards=forwards[:len(lefts)][:len(rights)][:len(lefts_acc)][:len(rights_acc)][:len(lefts_brake)][:len(rights_brake)][:len(brake)]
lefts=lefts[:len(forwards)]
rights=rights[:len(forwards)]
lefts_acc=lefts_acc[:len(forwards)]
rights_acc=rights_acc[:len(forwards)]
lefts_brake=lefts_brake[:len(forwards)]
rights_brake=rights_brake[:len(forwards)]
brake=brake[:len(forwards)]

final_data=forwards+lefts+rights+lefts_acc+rights_acc+lefts_brake+rights_brake+brake

