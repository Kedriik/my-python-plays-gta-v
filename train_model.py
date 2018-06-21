# -*- coding: utf-8 -*-
"""
Created on Tue May 15 18:57:50 2018

@author: Kedrowsky
"""
# =============================================================================
# from tensorflow.python.client import device_lib
# 
# def list_devices():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos]
# 
# list_devices()
# =============================================================================
import numpy as np
import tensorflow as tf
from alexnet import alexnet
import tflearn
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]



WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 8
#MODEL_NAME = 'pygta5-car-{}-{}-{}-epochs.model'.format(LR,'alexnetv2',EPOCHS)
MODEL_NAME = "model0"
train_data = np.load('training_data.v2.npy')
#with tf.device('/device:GPU:0'):
 # tf.reset_default_graph()
  #  sess = tf.InteractiveSession()
    
    
 #   sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    
train = train_data[:-500]
test = train_data[-500:]
X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1).astype(float)
Y=([i[1] for i in train])

test_x=np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1).astype(float)
test_y=([i[1] for i in test])

# =============================================================================
# model.fit({'input':X},{'targets':Y},n_epoch=EPOCHS,
#          validation_set=({'input':test_x},{'targets':test_y}),
#          snapshot_step=500,show_metric=True,run_id=MODEL_NAME)
# =============================================================================
#tflearn.init_graph(num_cores=4,gpu_memory_fraction=0.5)
#config = tf.ConfigProto(allow_soft_placement = True,log_device_placement=True)
#sess = tf.Session(config = config)


model = alexnet(WIDTH, HEIGHT, LR)
model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500,batch_size=512, show_metric=True, run_id=MODEL_NAME)
    
#tensorboard --logdir=foo:D:\Python projects\Python plays gta v\log

model.save(MODEL_NAME)

#variable_summaries()