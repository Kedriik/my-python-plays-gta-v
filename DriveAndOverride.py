import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
import os
from random import random

def keysToCategories(keys):
    if keys == [0,0,0,0]: #roll
        return [1,0,0,0,0,0,0,0,0]
    if keys == [1,0,0,0]: #roll-right
        return [0,1,0,0,0,0,0,0,0]
    if keys == [0,0,1,0]: #roll-left
        return [0,0,1,0,0,0,0,0,0]
    if keys == [0,1,0,0]: #acc
        return [0,0,0,1,0,0,0,0,0]
    if keys == [1,1,0,0]: #acc-right
        return [0,0,0,0,1,0,0,0,0]
    if keys == [0,1,1,0]: #acc-left
        return [0,0,0,0,0,1,0,0,0]
    if keys == [1,0,0,1]: #brake-right
        return [0,0,0,0,0,0,1,0,0]
    if keys == [0,0,1,1]: #brake-left
        return [0,0,0,0,0,0,0,1,0]
    if keys == [0,0,0,1]: #brake
        return [0,0,0,0,0,0,0,0,1]
    print("K to C: WARNING NON FOUND!")
    
def categoriesToKeys(categories):
    if categories == [1,0,0,0,0,0,0,0,0]: #roll
        return [0,0,0,0]
    if categories == [0,1,0,0,0,0,0,0,0]: #roll-right
        return [1,0,0,0]
    if categories == [0,0,1,0,0,0,0,0,0]: #roll-left
        return [0,0,1,0]
    if categories == [0,0,0,1,0,0,0,0,0]: #acc
        return [0,1,0,0]
    if categories == [0,0,0,0,1,0,0,0,0]: #acc-right
        return [1,1,0,0]
    if categories == [0,0,0,0,0,1,0,0,0]: #acc-left
        return [0,1,1,0]
    if categories == [0,0,0,0,0,0,1,0,0]: #brake-right
        return [1,0,0,1]
    if categories == [0,0,0,0,0,0,0,1,0]: #brake-left
        return [0,0,1,1]
    if categories == [0,0,0,0,0,0,0,0,1]: #brake
        return [0,0,0,1]

    print("C to K: WARNING NON FOUND!")
        
def keys_to_output(keys):
    #[A,W,D,S]
    output=[0,0,0,0]    
    if 'A' in keys:
        output[0] = 1
    if 'W' in keys:
        output[1] = 1
    if 'D' in keys:
        output[2] = 1
    if 'S' in keys:
        output[3] = 1        
    return output

def RandomAgent(state, reward):
    categories =  [0,0,0,0,0,0,0,0,0]
    for i in range(len(categories)):
        categories[i] = random()
    #print(categories)
    categories = np.array(categories)
    #print(categories)
    #print(np.argmax(categories))
    max_index = np.argmax(categories)
    for i in range(len(categories)):
        if i == max_index:
            categories[i] = 1
        else:
            categories[i] = 0
            
    print(categories)
    keys = categoriesToKeys(categories.tolist())
    print(keys)
            
def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
        
    last_time = time.time()
    start_time = time.time()
    collectData=False
    clickFlag=False
    print("Press Z to start collecting data")
    reward = 0
    while True:
        state = grab_screen(region=(0,40,800,640))
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state, (400,300))
        state = state.reshape(1,300,400)
        
        cv2.imshow('',state[0])
        if cv2.waitKey(25) & 0xFF==ord('q'):
            cv2.destroyAllWindows()
            break
        
        
        keys   = key_check()
        output = keys_to_output(keys)
        #print(output)
        if 'Z' in keys and clickFlag == False:
            clickFlag = True
            if collectData == True:
                collectData = False
                end_time = time.time()
                reward = end_time - start_time
                print('On time based reward:',reward)
            else:
                collectData = True
                start_time = time.time()
                print('Training started')
        elif 'Z' not in keys and clickFlag == True:
            clickFlag = False
            
                
        if 'X' in keys:
            print ('collecting data stopped')
            break
            
        #print('Loop took {} seconds'.format(time.time()  - last_time))
        last_time = time.time()
            
if __name__ == '__main__':
    main()