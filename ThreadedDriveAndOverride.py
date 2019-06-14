import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
from directkeys import PressKey, ReleaseKey, W,A,S,D
from random import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
from playground import agentZ, agentY, agentU
from threading import Thread
from threading import Lock
from time import sleep
from math import sqrt, exp
capture_region = (0,40,800,640)
reshape_size = (300,400)
AGENT_NAME = 'TestAgent'
g_debug_vars = []
def keysToCategories(keys):
    if keys == [0,0,0,0]: #rollx
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
    return [1,0,0,0,0,0,0,0,0]
    
def categoriesToKeys(categories):
    
    if categories == [1,0,0,0,0,0,0,0,0]: #rollzz
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
    return [0,0,0,0]
        
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
def teach_agent_rt(agent, reward, gradients, sess):
    feed_dict = {}
    for var_index, gradient_placeholder in enumerate(agent.gradient_placeholders):
        feed_dict[gradient_placeholder] = gradients[var_index]*reward;
    
    sess.run(agent.training_op, feed_dict=feed_dict)
        
def m_multinomial(acc, steer):
    acc_i = np.random.multinomial(1,acc)
    steer_i = np.random.multinomial(1,steer)
    return acc_i, steer_i
def m_multinomial9(action):
    return np.random.multinomial(1,action)

gathered_categories = []
mutex = Lock()
gather_flag = True
def gather_input():
    global gathered_categories
    key_check()
    while(gather_flag):
        keys   = key_check()
        mutex.acquire(1)
        gathered_categories.append(keysToCategories(keys_to_output(keys)))
        mutex.release()
        sleep(0.1)
def normalize(v):
    m = 0
    for i in range(len(v)):
        m = m + v[i]*v[i]
    m = sqrt(m)
    for i in range(len(v)):
        v[i] = v[i] / m
    return v

def softmax(x):
    _sum = 0.
    for i in range(len(x)):
        _sum = _sum + exp(x[i])
    for i in range(len(x)):
        x[i] = exp(x[i]) / _sum
    return x

def harvest_input():
    global gathered_categories
    average_categories = np.array([0,0,0,0,0,0,0,0,0]).astype(float)
    mutex.acquire(1)
    for i in range(len(gathered_categories)):
        average_categories = np.array(average_categories) + np.array(gathered_categories[i]).astype(float)
    
    average_categories = softmax(average_categories)    
    
    gathered_categories.clear()
    mutex.release()
    return average_categories
teach_agent_flag = True
save_flag = False
def teach_agent():
    tf.reset_default_graph()
    
    path_to_agent = AGENT_NAME
    agent = agentU(0.1,(300,400,1),9,11)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        try:
            with open('{}.index'.format(path_to_agent),'r') as fh:
                print('Agent {} exists. Loading.'.format(AGENT_NAME))
                saver.restore(sess,path_to_agent)
                print('Loading complete')
        except FileNotFoundError:
            print('Agent {} doesnt exist.Creating.'.format(AGENT_NAME))
            
            init = tf.global_variables_initializer()
            sleep(1)
            sess.run(init)
            print('Creation complete')
        #saver = tf.train.Saver()xx
        while(teach_agent_flag):
            loop_start = time.time()
            state = get_state()
            harvested_input = harvest_input()
            #xxx
            feed_dict_learn={agent.state_in:[state],
                       agent.action_in:[harvested_input]}
            feed_dict_steer = {agent.state_in:[state]}
            #xxxxxxxxxxxxx
            #action = sess.run([agent.action_logits], feed_dict=feed_dicxxxt_steer)xxx
            _,loss = sess.run([agent.minimize, agent.loss], feed_dict = feed_dict_learn)
            print("Loss:", loss)
            loop_duration = time.time() - loop_start
            print("ML loop took:", loop_duration)
            #print("action:",action)
            if(loop_duration < 1.0):
                sleep(1 - loop_duration)
                
        saver.save(sess, path_to_agent)
    
def control():
    global teach_agent_flag
    global gather_flag        
    collectData = False
    clickFlag = False
    while True:
        time0 = time.time()
        #-----------------
        keys = []
        keys   = key_check()
        
        if 'Z' in keys and clickFlag == False:
            clickFlag = True
            if collectData == True:
                collectData = False
            else:
                collectData = True
                print('Training started')
        elif 'Z' not in keys and clickFlag == True:
            clickFlag = False
            
                
        if 'X' in keys:
            teach_agent_flag = False
            gather_flag = False
            print ('collecting data stopped')
            break
        
        if 'T' in keys and collectData == False:
            print ('Saving agent, please wait')
            #saver.save(sess, '/tmp/model{}.ckpt'.format(teach_count))
            print('Saved.');
   
        time1 = time.time()
        delta_time = time1 - time0
        
def main():
    tf.reset_default_graph()
    gather_input_thread = Thread(target = gather_input)
    control_thread = Thread(target = control)
    gather_input_thread.start()
    control_thread.start() 
    
    teach_agent()
        
    gather_input_thread.join()
    control_thread.join()
        
if __name__ == '__main__':
    main()