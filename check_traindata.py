# -*- coding: utf-8 -*-
"""
Created on Mon May 14 20:59:50 2018

@author: Kedrowsky
"""
import numpy as np
import cv2
#train_data = np.load('training_data_npy.npy')
#array=[1,2,3,4]
#array1=array[::-1]
for data in final_data:
    img=data[0]
    #img1=np.flip(img, 1)
    choice=data[1]
    #cv2.imshow('fliped',img1)
    cv2.imshow('original',img)
    print(choice)
    if cv2.waitKey(25) & 0xFF==ord('q'):
        cv2.destroyAllWindows()
        break