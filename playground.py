import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
import os
import tensorflow as tf
from tensorflow.contrib import slim
class agentZ():
    def __init__(self, lr, s_size,a_size,h_size):
        self.state_in = tf.placeholder(shape=[None]+list(s_size),dtype=tf.float32)
        conv1         = slim.conv2d(self.state_in,num_outputs = 16, kernel_size = [4,4])
        max_pool1     = slim.max_pool2d(conv1,[2, 2])
        hidden        = slim.fully_connected(max_pool1,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.flatten  = slim.flatten(hidden)
        self.logits   = slim.fully_connected(self.flatten,num_outputs=a_size,activation_fn=tf.nn.sigmoid,biases_initializer=None)
        self.action   = tf.multinomial(tf.log(self.output),num_samples = 1)

class agentY():
    def __init__(self,lr,s_size,a_size,h_size):
        self.state_in = tf.placeholder(shape = [None]+list(s_size),dtype=tf.float32)
        conv1         = tf.layers.conv2d(self.state_in,32,4,strides=(4, 4))
        max_pool1     = tf.layers.max_pooling2d(conv1,32,4)
        flatten       = tf.layers.flatten(max_pool1)
        hidden        = tf.layers.dense(flatten,4096,activation=tf.nn.tanh)
        
        
        hidden_acc              = tf.layers.dense(hidden,2048, activation=tf.nn.elu)
        self.logits_acc         = tf.layers.dense(hidden_acc,1, activation=tf.nn.sigmoid)
        acc_press_or_not        = tf.concat(axis=1, values=[self.logits_acc, 1 - self.logits_acc])
        self.acc_action         = tf.multinomial(tf.log(acc_press_or_not), num_samples=1)
        self.y_acc              = 1. - tf.to_float(self.acc_action)        
        
        hidden_brake            = tf.layers.dense(hidden,2048, activation=tf.nn.elu)
        self.logits_brake       = tf.layers.dense(hidden_brake,1, activation=tf.nn.sigmoid)
        brake_press_or_not      = tf.concat(axis=1, values=[self.logits_brake, 1 - self.logits_brake])
        self.brake_action       = tf.multinomial(tf.log(brake_press_or_not), num_samples=1)
        self.y_brake            = 1. - tf.to_float(self.brake_action)        
        
        hidden_steer_left       = tf.layers.dense(hidden,2048, activation=tf.nn.elu)
        self.logits_left        = tf.layers.dense(hidden_steer_left,1, activation=tf.nn.sigmoid)
        left_press_or_not       = tf.concat(axis=1, values=[self.logits_left, 1 - self.logits_left])
        self.left_action        = tf.multinomial(tf.log(left_press_or_not), num_samples=1)
        self.y_left             = 1. - tf.to_float(self.left_action)        
        
        hidden_steer_right      = tf.layers.dense(hidden,2048, activation=tf.nn.elu)
        self.logits_right       = tf.layers.dense(hidden_steer_right,1, activation=tf.nn.sigmoid)
        right_press_or_not      = tf.concat(axis=1, values=[self.logits_right, 1 - self.logits_right])
        self.right_action       = tf.multinomial(tf.log(right_press_or_not), num_samples=1)
        self.y_right            = 1. - tf.to_float(self.right_action)        
        

class agentZ():
    def __init__(self,lr,s_size,a_size,h_size):
        self.state_in = tf.placeholder(shape = [None]+list(s_size),dtype=tf.float32)
        conv1         = tf.layers.conv2d(self.state_in,32,4,strides=(4, 4))
        max_pool1     = tf.layers.max_pooling2d(conv1,32,4)
        flatten       = tf.layers.flatten(max_pool1)
        hidden        = tf.layers.dense(flatten,4096,activation=tf.nn.tanh)
        
        
        hidden_acc            = tf.layers.dense(hidden,2048, activation=tf.nn.relu)
        self.acc_action       = tf.layers.dense(hidden_acc,3, activation=tf.nn.softmax)
        #self.acc_output       = tf.to_float(tf.equal(tf.reduce_max(self.dense_acc_output, axis=1), self.dense_acc_output))
        #self.acc_logits       = tf.argmax(self.acc_output, axis=1)
        #self.acc_action       = tf.multinomial(tf.log(self.dense_acc_output),1)
        
        hidden_steer          = tf.layers.dense(hidden,2048, activation=tf.nn.relu)
        self.steer_action     = tf.layers.dense(hidden_steer,3, activation=tf.nn.softmax)
        #self.steer_output       = tf.to_float(tf.equal(tf.reduce_max(self.dense_steer_output, axis=1), self.dense_steer_output))
        #self.logits             = tf.argmax(self.steer_output, axis=1)
        #self.steer_action       = tf.multinomial(tf.log(self.dense_steer_output),1)
        self.acc_in           = tf.placeholder(shape =[None,3],dtype=tf.float32) 
        self.steer_in         = tf.placeholder(shape =[None,3],dtype=tf.float32) 
        
        cross_entropy_acc     = tf.nn.softmax_cross_entropy_with_logits(labels=self.acc_in, logits=self.acc_action)
        cross_entropy_steer   = tf.nn.softmax_cross_entropy_with_logits(labels=self.steer_in, logits=self.steer_action)
        optimizer             = tf.train.AdamOptimizer(lr)

        grads_and_vars = optimizer.compute_gradients(cross_entropy_acc)
        print(grads_and_vars)
        gradients = [grad for grad, variable in grads_and_vars]
        gradient_placeholders = []
        grads_and_vars_feed = []
        for grad, variable in grads_and_vars:
            gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
            gradient_placeholders.append(gradient_placeholder)
            grads_and_vars_feed.append((gradient_placeholder, variable))
        training_op = optimizer.apply_gradients(grads_and_vars_feed)
        

        
tf.reset_default_graph()
testAgent = agentZ(0.1,(300,400,1),9,11)  
     
class lesson2agent():
    def __init__(self, lr, s_size,a_size,h_size):
        #These lines established the feed-forward part of the network.
        #The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        #print("Output:",self.output)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure.
        #We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        #print("output[0]:",self.output[0])
        #print("output[1]:",self.output[1])
        #print("action_holder:",self.action_holder)
        #print("indexes:",self.indexes)
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
        #print("responsible outputs:",self.responsible_outputs)
        #print("reward holder:",self.reward_holder)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))
myAgent = lesson2agent(lr=1e-2,s_size=4,a_size=2,h_size=8) #Load the agent.

class lesson2agent2():
    def __init__(self, lr, s_size,a_size,h_size):
        self.state_in= tf.placeholder(shape=[None]+list(s_size),dtype=tf.float32)
        conv1 = slim.conv2d(self.state_in,num_outputs = 16, kernel_size = [4,4])
        max_pool1 = slim.max_pool2d(conv1,[2, 2])
        hidden = slim.fully_connected(max_pool1,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        flatten = slim.flatten(hidden)
        self.output = slim.fully_connected(flatten,num_outputs=a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        #print("Output:",self.output)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure.
        #We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        #print("output[0]:",self.output[0])
        #print("output[1]:",self.output[1])
        #print("action_holder:",self.action_holder)
        #print("indexes:",self.indexes)
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
        #print("responsible outputs:",self.responsible_outputs)
        #print("reward holder:",self.reward_holder)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))
# =============================================================================
# def keysToCategories(keys):
#     if keys == [0,0,0,0]: #roll
#         return [1,0,0,0,0,0,0,0,0]
#     if keys == [1,0,0,0]: #roll-right
#         return [0,1,0,0,0,0,0,0,0]
#     if keys == [0,0,1,0]: #roll-left
#         return [0,0,1,0,0,0,0,0,0]
#     if keys == [0,1,0,0]: #acc
#         return [0,0,0,1,0,0,0,0,0]
#     if keys == [1,1,0,0]: #acc-right
#         return [0,0,0,0,1,0,0,0,0]
#     if keys == [0,1,1,0]: #acc-left
#         return [0,0,0,0,0,1,0,0,0]
#     if keys == [1,0,0,1]: #brake-right
#         return [0,0,0,0,0,0,1,0,0]
#     if keys == [0,0,1,1]: #brake-left
#         return [0,0,0,0,0,0,0,1,0]
#     if keys == [0,0,0,1]: #brake
#         return [0,0,0,0,0,0,0,0,1]
#     print("WARNING NON FOUND!")
#     
# resized = [400,300]
# array = None
# while True:
#     screen = grab_screen(region=(0,40,800,640))
#     screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
#     screen = cv2.resize(screen,(resized[0],resized[1]))
#     if array is None or array.shape[0] < 60:
#         if  array is None:
#             array = screen.reshape(1,resized[1],resized[0])
#         else:
#             array = np.append(array, screen.reshape(1,resized[1],resized[0]),0)
#     else:
#         array = np.delete(array,0,0)
#         array = np.append(array, screen.reshape(1,resized[1],resized[0]),0)
#         
#     cv2.imshow('',array[0])
#     if cv2.waitKey(25) & 0xFF==ord('q'):
#         cv2.destroyAllWindows()
#         break
#    
# =============================================================================
# =============================================================================
# while True:
#     screen = grab_screen(region=(0,40,800,640))
#     screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
#     screen = cv2.resize(screen, (400,300))
#     array = screen.reshape(300,400,1)
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