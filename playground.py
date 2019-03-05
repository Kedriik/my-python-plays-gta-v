import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
import os


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
    print("WARNING NON FOUND!")
    
resized = [400,300]
array = None
while True:
    screen = grab_screen(region=(0,40,800,640))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen = cv2.resize(screen,(resized[0],resized[1]))
    if array is None or array.shape[0] < 60:
        if  array is None:
            array = screen.reshape(1,resized[1],resized[0])
        else:
            array = np.append(array, screen.reshape(1,resized[1],resized[0]),0)
    else:
        array = np.delete(array,0,0)
        array = np.append(array, screen.reshape(1,resized[1],resized[0]),0)
        
    cv2.imshow('',array[0])
    if cv2.waitKey(25) & 0xFF==ord('q'):
        cv2.destroyAllWindows()
        break
   
# =============================================================================
# while True:
#     screen = grab_screen(region=(0,40,800,640))
#     screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
#     screen = cv2.resize(screen, (400,300))
#     array = screen.reshape(1,300,400)
#         
#     cv2.imshow('',array[0])
#     if cv2.waitKey(25) & 0xFF==ord('q'):
#         cv2.destroyAllWindows()
#         break
# =============================================================================
        
# =============================================================================
# for i in range(60):
#     #screen = grab_screen(region=(0,40,800,640))
#     #screen=cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('',array[i])
#     if cv2.waitKey(25) & 0xFF==ord('q'):
#         cv2.destroyAllWindows()
#         break
# =============================================================================

#cv2.destroyAllWindows()