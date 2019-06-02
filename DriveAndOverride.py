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
g_debug_vars = []
def keysToCategories(keys):
    if keys == [0,0,0,0]: #rollzzt
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

    #print("C to K: WARNING NON FOUND!")zz
        
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
def show_frames(frames):
    for frame in frames:
        cv2.imshow('',frame)
        if cv2.waitKey(25) & 0xFF==ord('q'):
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
        
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

def teach_agent(agent, all_rewards, all_gradients,sess):
    rewards = np.array(discount_and_normalize_rewards(all_rewards,0.99))
    test = []
    feed_dict = {}

    
    for var_index, gradient_placeholder in enumerate(agent.gradient_placeholders):
        mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                  for game_index, rewards in enumerate(all_rewards)
                                      for step, reward in enumerate(rewards)], axis=0)
        
        feed_dict[gradient_placeholder] = mean_gradients
    ret = sess.run(agent.training_op, feed_dict=feed_dict)  
    print(ret)
    
def m_multinomial(acc, steer):
    acc_i = np.random.multinomial(1,acc)
    steer_i = np.random.multinomial(1,steer)
    return acc_i, steer_i
def m_multinomial9(action):
    return np.random.multinomial(1,action)
def main():        
    collectData = False
    clickFlag = False
    do_action = False
    teach_count = 0
    
    # Add ops to save and restore all the variables.zzt
    #saver = tf.train.Saver()

    last_state = get_state()    
    tf.reset_default_graph() #Clear the Tensorflow graph.
    myAgent = agentY(0.1,(300,400,1),9,11)
    history = []
    delta_time = 0
    state = last_state
    init = tf.global_variables_initializer()
    reward = 0 
    key_check()
# Launch the tensorflow graphzzzzz
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        print("Loop starts")
        print("Press Z to start collecting data")
        all_rewards         = []
        all_gradients       = []
        current_rewards     = []
        current_gradients   = []
        while True:
            time0 = time.time()
            #-----------------zzx
            if collectData:
                state = get_state()
                action_out, gradients = sess.run([myAgent.action_out,myAgent.gradients], feed_dict={myAgent.state_in:[state]})
                
                print("Action:",action_out[0][0])
                print("Gradients:", gradients)
                
                
                #action = m_multinomial9(action[0][0])
                if do_action:
                    releaseKeys();
                    update_pressed_keys(categoriesToKeys(action))
                    
                reward = reward + delta_time
                current_rewards.append(reward)
                current_gradients.append(gradients)
                #g_debug_vars.append(myAgent.gradients)
                #-----------------------------zzzzzztzztzzzzzztzzztzt
            keys = []
            keys   = key_check()
            
            if 'Z' in keys and clickFlag == False:
                clickFlag = True
                if collectData == True:
                    collectData = False
                    releaseKeys()
                    print('Time:',reward)
                    all_rewards.append(current_rewards)
                    all_gradients.append(current_gradients)
                    del current_rewards
                    del current_gradients
                    current_rewards     = []
                    current_gradients   = []
                else:
                    collectData = True
                    reward = 0
                    print('Training started')
            elif 'Z' not in keys and clickFlag == True:
                clickFlag = False
                
                    
            if 'X' in keys:
                print ('collecting data stopped')
                break
            
            if 'T' in keys and collectData == False:
                print ('Teaching agent, please wait')
                teach_agent(myAgent, all_rewards, all_gradients,sess) 
                #save_path = saver.save(sess, '/tmp/model{}.ckpt'.format(teach_count))
                    
                del all_rewards
                del all_gradients
                all_gradients = []
                all_rewards = []
       
            time1 = time.time()
            delta_time = time1 - time0
            
if __name__ == '__main__':
    main()