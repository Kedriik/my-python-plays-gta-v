# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 19:38:50 2018

@author: Kedrowsky
"""

import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
from my_alexnet import alexnet
from directkeys import PressKey, ReleaseKey, W,A,S,D
import tensorflow as tf
tf.reset_default_graph()
import os
WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 8
#MODEL_NAME = "models\current_model\model0.model" #'pygta5-car-{}-{}-{}-epochs.model'.format(LR,'alexnetv2',EPOCHS)
#MODEL_NAME = "main_loop_model.model"#"models\main_loop_model"
MODEL_NAME="main_loop_model_extended_1"
    
def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(A)
    PressKey(W)
    ReleaseKey(D)

def right():
    PressKey(D)
    PressKey(W)
    ReleaseKey(A)

def brake():
    PressKey(S)
    ReleaseKey(W)

model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

accelerationCounter=0
def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
        
    last_time = time.time()
    paused= False
    clickFlag=False
    while True:
        if not paused:
            screen = grab_screen(region=(0,40,800,640))
            screen=cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen=cv2.resize(screen,(160,120))
    
            last_time = time.time()
            prediction=model.predict([screen.reshape(WIDTH, HEIGHT,1)])[0]
            moves=list(np.around(prediction))
            print(moves, prediction)
# =============================================================================
#     if 'A' in keys:
#         output[0] = 1
#     if 'W' in keys:
#         output[1] = 1
#     if 'D' in keys:
#         output[2] = 1
#     if 'S' in keys:
#         output[3] = 1
# =============================================================================
            ReleaseKey(A)
       #     ReleaseKey(W)
            ReleaseKey(D)
       #     ReleaseKey(S)
            accT=0.0
            accFactor=5;
            prediction[1]*=accFactor
            brakeT=0.95
            leftT=0.1
            #rightMultFactor=0.8;
            #prediction[2]*=rightMultFactor
            rightT=0.1
            if prediction[0] > prediction[2]:
                if prediction[0] > leftT:
                    ReleaseKey(D)
                    PressKey(A)
            else:
                if prediction[2] > rightT:
                    ReleaseKey(A)
                    PressKey(D)
                    
            if prediction[1] > prediction[3]:
                if prediction[1] > accT:
                    ReleaseKey(S)
                    #PressKey(W)
                    accelerationCounter=10
            else:
                if prediction[3] > brakeT:
                    ReleaseKey(W)
                    PressKey(S)
                    accelerationCounter=0
                    

            if(accelerationCounter>0):
                PressKey(W)
                accelerationCounter-=1
            else:
                ReleaseKey(W)
                
            ReleaseKey(A)
            ReleaseKey(W)
            ReleaseKey(D)
            ReleaseKey(S)
            
        keys=key_check()
        if 'T' in keys and clickFlag==False:
            clickFlag=True
            if paused==True:
                paused=False
                print('resumed')
            else:
                paused=True
                ReleaseKey(W)
                ReleaseKey(S)
                ReleaseKey(A)
                ReleaseKey(D)
                print('paused')
        elif 'T' not in keys and clickFlag==True:
            clickFlag=False
        
        if 'X' in keys:
            break

            
            
if __name__ == '__main__':
    main()