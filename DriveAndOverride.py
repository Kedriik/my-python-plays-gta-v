import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
from directkeys import PressKey, ReleaseKey, W,A,S,D
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
def update_pressed_keys(keys):
    ReleaseKey(W)
    ReleaseKey(S)
    ReleaseKey(A)
    ReleaseKey(D)
    if keys[0] == 1:
        PressKey(A)
    if keys[1] == 1:
        PressKey(W)
    if keys[2] == 1:
        PressKey(D)
    if keys[3] == 1:
        PressKey(S)
        
def releaseKeys():
    ReleaseKey(W)
    ReleaseKey(S)
    ReleaseKey(A)
    ReleaseKey(D)
    
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
            
    #print(categories)
    keys = categoriesToKeys(categories.tolist())
    #print(keys)
    return keys
            
def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
        

    collectData=False
    clickFlag=False
    print("Press Z to start collecting data")
    reward = 0
    capture_region = (0,40,800,640)
    reshape_size = (300,400)
    last_state = grab_screen(region = capture_region)
    last_state = cv2.cvtColor(last_state, cv2.COLOR_BGR2GRAY)
    last_state = cv2.resize(last_state,(reshape_size[1],reshape_size[0]))
    last_state = last_state.reshape(1,reshape_size[0],reshape_size[1])
    
    history = []
    delta_time = 0
    while True:
        time0 = time.time()
        #make action
        action_keys = RandomAgent(1,1)
        if collectData == True:
            update_pressed_keys(action_keys)
        action = keysToCategories(action_keys)
        #-----------zzzzzzzzzzzzzzzzzz
        keys   = key_check()
        
        if 'Z' in keys and clickFlag == False:
            clickFlag = True
            if collectData == True:
                collectData = False
                releaseKeys()
                print('On time based reward:',reward)
                history_np = np.array(history)
                frames = history_np[:,0]
                for frame in frames:
                    cv2.imshow('',frame[0])
                    if cv2.waitKey(25) & 0xFF==ord('q'):
                        cv2.destroyAllWindows()
                        break
                cv2.destroyAllWindows()   
                #update agent
            else:
                collectData = True
                #start_time = time.time()
                reward = 0
                print('Training started')
        elif 'Z' not in keys and clickFlag == True:
            clickFlag = False
            
                
        if 'X' in keys:
            print ('collecting data stopped')
            break
        reward = reward + delta_time
        
        #acquire new environment state 
        state = grab_screen(region = capture_region)
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state, (reshape_size[1],reshape_size[0]))
        state = state.reshape(1,reshape_size[0],reshape_size[1])
        #-----------------------------
        history.append([last_state,action,reward,state])
        last_state = state        
# =============================================================================
#         cv2.imshow('',state[0])
#         if cv2.waitKey(25) & 0xFF==ord('q'):
#             cv2.destroyAllWindows()
#             break
# =============================================================================
        #print('Loop took {} seconds'.format(time.time()  - last_time))        
        time1 = time.time()
        delta_time = time1 - time0
            
if __name__ == '__main__':
    main()