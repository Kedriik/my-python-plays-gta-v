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

# =============================================================================
# train_data = np.load('training_data_npy.npy')
# print(len(train_data))
# df=pd.DataFrame(train_data)
# print(df.head())
# print(Counter(df[1].apply(str)))
# =============================================================================

lefts=[]
rights=[]
forwards=[]
lefts_acc=[]
rights_acc=[]
lefts_brake=[]
rights_brake=[]
brake=[]
roll=[]

shuffle(glued_train_data)

for data in glued_train_data:
    img =data[0]
    choice=data[1]
    #print(choice)
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
        
# =============================================================================
#     if choice==[1,0,0,1]:
#         lefts_brake.append([img, choice])
#         
#     if choice==[0,0,1,1]:
#         rights_brake.append([img, choice])
# =============================================================================

    if choice==[0,0,0,1]:
        brake.append([img, choice])
        
    if choice==[0,0,0,0]:
        roll.append([img, choice])
final_data=[]
forwards=forwards[:len(lefts)][:len(rights)][:len(lefts_acc)][:len(rights_acc)][:len(brake)][:len(roll)]
lefts=lefts[:len(forwards)]
rights=rights[:len(forwards)]
lefts_acc=lefts_acc[:len(forwards)]
rights_acc=rights_acc[:len(forwards)]
lefts_brake=lefts_brake[:len(forwards)]
rights_brake=rights_brake[:len(forwards)]
brake=brake[:len(forwards)]

final_data=forwards+lefts+rights+lefts_acc+rights_acc+brake

shuffle(final_data)
final_data_length=len(final_data)
for i in range(final_data_length):
    data=final_data[i]
    reversedImage=np.flip(data[0], 1)
    originalInput=data[1]
    reversedInput=[0,0,0,0]
    reversedInput[0]=originalInput[2]
    reversedInput[2]=originalInput[0]
    reversedInput[1]=originalInput[1]
    reversedInput[3]=originalInput[3]
    final_data.append([reversedImage, reversedInput])
    
shuffle(final_data)
    



