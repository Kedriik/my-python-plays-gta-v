import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
import os

def show_captured_screen():
    while True:
        screen=grab_screen(region=(0,40,800,640))
        screen=cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        cv2.imshow('ScreenLive',screen)
        if cv2.waitKey(25) & 0xFF==ord('q'):
            cv2.destroyAllWindows()
            break
        
def keys_to_output(keys):
    #[A,W,D]
    # [S]
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

file_name='main_loop1.npy'

if os.path.isfile(file_name):
    print('File exists, loading precious data!')
    training_data=list(np.load(file_name))
else:
    print('File does not exist, starting fresh')
    training_data=[]
    

        


def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
        
    last_time = time.time()
    
    collectFlag=False
    clickFlag=False
    print("Press Z to start collecting data")
    while True:
        screen = grab_screen(region=(0,40,800,640))
        screen=cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen=cv2.resize(screen,(160,120))
        keys=key_check()
        output=keys_to_output(keys)
        if collectFlag:
            training_data.append([screen,output])
 #      print('Frame took {} seconds'.format(time.time()-last_time))
       # last_time = time.time()
        if len(training_data)%1000 ==0 and collectFlag==True:
            print(len(training_data))
        if 'B' in keys:
            collectFlag=False
            last_time = time.time()
            print('starting saving data length:{}'.format(len(training_data)))
            np.save(file_name,training_data)
            print('Saving data took {} seconds'.format(time.time()-last_time))
            print("Data saved. Press Z to resume collecting")
            
        if 'Z' in keys and clickFlag==False:
            clickFlag=True
            if collectFlag==True:
                collectFlag=False
                print('collecting data pauxsed')
            else:
                collectFlag=True
                print('collecting data resumed')
        elif 'Z' not in keys and clickFlag==True:
            clickFlag=False
            
                
        if 'X' in keys:
            print ('collecting data stopped')
            break
            
            
if __name__ == '__main__':
    main()