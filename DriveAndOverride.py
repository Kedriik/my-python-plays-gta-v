import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
from directkeys import PressKey, ReleaseKey, W,A,S,D
from random import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
from playground import agentZ

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
gamma = 0.99
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

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

class agent():
    def __init__(self, lr, s_size,a_size,h_size):
        #These lines established the feed-forward part of the network.
        #The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size[0],s_size[1],s_size[2]],dtype=tf.float32)
        print("state in:",self.state_in)
        conv1 = slim.conv2d(self.state_in,num_outputs = 64, kernel_size = [4,4])
        print("conv1:",conv1)
        max_pool1 = slim.max_pool2d(conv1,[2, 2])
        print("max_pool1:",max_pool1)
        hidden = slim.fully_connected(max_pool1,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        print("hidden:",hidden)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        print("output:",self.output)
        self.chosen_action = tf.argmax(self.output,1)

def main():        
    collectData = False
    clickFlag = False
    print("Press Z to start collecting data")
    reward = 0
    capture_region = (0,40,800,640)
    reshape_size = (300,400)
    last_state = grab_screen(region = capture_region)
    last_state = cv2.cvtColor(last_state, cv2.COLOR_BGR2GRAY)
    last_state = cv2.resize(last_state,(reshape_size[1],reshape_size[0]))
    last_state = last_state.reshape(reshape_size[0],reshape_size[1],1)
    
    tf.reset_default_graph() #Clear the Tensorflow graph.
    myAgent = agentZ(0.1,(300,400,1),9,11)
    history = []
    delta_time = 0
    state = last_state
    init = tf.global_variables_initializer()
    reward = 0 #[0,0,0,0,0,0,0,0,0]
# Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        gradBuffer = sess.run(tf.trainable_variables())
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0
        print("Loop starts")
        while True:
            time0 = time.time()
            #make action
            action_keys = sess.run(myAgent.output,feed_dict={myAgent.state_in:[state]})
            max_id = np.argmax(action_keys)
            #    update_pressed_keys(action_keys)
            action = [0,0,0,0,0,0,0,0,0]
            action[max_id] = 1
            agent_keys = categoriesToKeys(action)
# =============================================================================
#             if(collectData):
#                 #Agent's decision
#                 update_pressed_keys(agent_keys)
# =============================================================================
            #-----------
            if collectData:
                policy = [1,1,1,1,1,1,1,1,1]
                reward = reward + delta_time*np.array(policy)
                state = grab_screen(region = capture_region)
                state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
                state = cv2.resize(state, (reshape_size[1], reshape_size[0]))
                state = state.reshape(reshape_size[0], reshape_size[1],1)
                #-----------------------------
                history.append([last_state,action,reward,state]) 
                last_state = state        
            keys = []
            keys   = key_check()
            
            if 'Z' in keys and clickFlag == False:
                clickFlag = True
                if collectData == True:
                    collectData = False
                    releaseKeys()
                    print('On time based reward:',reward)
                    history = np.array(history)
                    frames = np.array(history[:,0])
                    rewards = np.array(discount_rewards(history[:,2]))
                    actions = np.array(history[:,1])
# =============================================================================
#                     i = 0
#                     for frame in frames:
#                         print(actions[i])
#                         i = i + 1
#                         cv2.imshow('',frame.reshape((reshape_size[0],reshape_size[1],1)))
#                         if cv2.waitKey(25) & 0xFF==ord('q'):
#                             cv2.destroyAllWindows()
#                             break
#                     cv2.destroyAllWindows()
# =============================================================================
                    del history
                    history = []
                    #update agent
                else:
                    collectData = True
                    reward = 0
                    print('Training started')
            elif 'Z' not in keys and clickFlag == True:
                clickFlag = False
                
                    
            if 'X' in keys:
                print ('collecting data stopped')
                break
       
            time1 = time.time()
            delta_time = time1 - time0
            
if __name__ == '__main__':
    main()