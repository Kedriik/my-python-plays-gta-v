import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
from directkeys import PressKey, ReleaseKey, W,A,S,D
from random import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
from playground import agentZ, agentY

capture_region = (0,40,800,640)
reshape_size = (300,400)
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

    #print("C to K: WARNING NON FOUND!")
        
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
        PressKey(W)
    if keys[1] == 1:
        PressKey(S)
    if keys[2] == 1:
        PressKey(A)
    if keys[3] == 1:
        PressKey(D)
        
def releaseKeys():
    ReleaseKey(W)
    ReleaseKey(S)
    ReleaseKey(A)
    ReleaseKey(D)

gamma = 0.99
def _discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

def get_state():
    state = grab_screen(region = capture_region)
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.resize(state,(reshape_size[1],reshape_size[0]))
    state = state.reshape(reshape_size[0],reshape_size[1],1)
    return state

def teach_agent(agent, history,sess):
    history = np.array(history)
    frames  = np.array(history[:,0])
    rewards = np.array(discount_rewards(history[:,3],0.99))
    accs = np.array(history[:,1])
    steers = (history[:,2])
    #steers = [np.array[steer] for steer in steers]
    #=([steer for steer in steers])
    #print(Y)
    steers  = np.array([steer for steer in steers]).reshape(-1,3).astype(float)
    accs    = np.array([acc for acc in accs]).reshape(-1,3).astype(float)
    frames  = np.array([i for i in frames]).reshape(-1,300,400,1).astype(float)
    #print(X)
      
    ret = sess.run(agent.steer_in, feed_dict={agent.acc_in:accs, agent.steer_in:steers, agent.state_in:frames})
def m_multinomial(acc, steer):
    acc_i = np.random.multinomial(1,acc)
    steer_i = np.random.multinomial(1,steer)
    return acc_i, steer_i
def main():        
    collectData = False
    clickFlag = False
    last_state = get_state()    
    tf.reset_default_graph() #Clear the Tensorflow graph.
    myAgent = agentZ(0.1,(300,400,1),9,11)
    history = []
    delta_time = 0
    state = last_state
    init = tf.global_variables_initializer()
    reward = 0 
# Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        print("Loop starts")
        print("Press Z to start collecting data")
        while True:
            time0 = time.time()
            #accelerate_or_brake
            acc,steer = sess.run([myAgent.acc_action,myAgent.steer_action],
                                 feed_dict={myAgent.state_in:[state]})
            #action = [acc[0][0],brake[0][0],left[0][0],right[0][0]]
            acc, steer = m_multinomial(acc[0],steer[0])
            
            acc = np.array([acc[0],acc[1],acc[2]])
            steer = np.array([steer[0],steer[1],steer[2]])
            #print(acc)
            releaseKeys();
            #if  collectData:
            #    print(action[0],action[1])
                
                #Agent's decision
                #update_pressed_keys(action)
            #-----------xxzzxxzzzzzzxzzzxxzzzzzzxzzzxxzzzzzzz
            if collectData:
                reward = reward + delta_time
                state = grab_screen(region = capture_region)
                state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
                state = cv2.resize(state, (reshape_size[1], reshape_size[0]))
                state = state.reshape(reshape_size[0], reshape_size[1],1)
                #-----------------------------
                history.append([last_state,acc, steer,reward,state]) 
                last_state = state        
            keys = []
            keys   = key_check()
            
            if 'Z' in keys and clickFlag == False:
                clickFlag = True
                if collectData == True:
                    collectData = False
                    releaseKeys()
                    print('On time based reward:',reward)
                    teach_agent(myAgent, history,sess)                    

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