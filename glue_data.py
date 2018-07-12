# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 20:20:37 2018

@author: Kedrowsky
"""


import time
start_time=time.time()
import cv2
import numpy as np
glued_train_data=[]
train_data=np.load('main_loop.npy')
for data in train_data:
    img =data[0]
    choice=data[1]
    glued_train_data.append([img, choice])
    
    
train_data=np.load('main_loop1.npy')
for data in train_data:
    img =data[0]
    choice=data[1]
    glued_train_data.append([img, choice])
    
    
train_data=np.load('main_loop2.npy')
for data in train_data:
    img =data[0]
    choice=data[1]
    glued_train_data.append([img, choice])
    
train_data=np.load('main_loop3.npy')
for data in train_data:
    img =data[0]
    choice=data[1]
    glued_train_data.append([img, choice])
    
    
del train_data
#np.save('glued_train_data.npy',glued_train_data)

print('Script took {} seconds to execute'.format(time.time()-start_time))