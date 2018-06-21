# -*- coding: utf-8 -*-
"""
Created on Mon May 14 20:59:50 2018

@author: Kedrowsky
"""
import numpy as np
import cv2
train_data = np.load('training_data_npy.npy')

for data in train_data:
    img=data[0]
    choice=data[1]
    cv2.imshow('test',img)
    print(choice)
    if cv2.waitKey(25) & 0xFF==ord('q'):
        cv2.destroyAllWindows()
        break