# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 12:25:37 2018

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
from my_alexnet import alexnet
import tflearn
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]



WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 30
#MODEL_NAME = 'pygta5-car-{}-{}-{}-epochs.model'.format(LR,'alexnetv2',EPOCHS)
MODEL_NAME = "main_loop_model_extended_1"
#train_data = np.load('training_data.v2.npy')
#with tf.device('/device:GPU:0'):
 # tf.reset_default_graph()
  #  sess = tf.InteractiveSession()
    
    
 #   sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

data_per_iteration=50000
model = alexnet(WIDTH, HEIGHT, LR)

for i in range(int((len(final_data)/data_per_iteration))):
    current_data=[]
    if (i+1)*data_per_iteration < len(final_data):
        current_data=final_data[i*data_per_iteration:(i+1)*data_per_iteration]
    else:
        current_data=final_data[-(len(final_data)%data_per_iteration)]
    
    train = current_data[:-1000]
    test = current_data[-1000:]
    X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1).astype(float)
    Y=([i[1] for i in train])
    
    test_x=np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1).astype(float)
    test_y=([i[1] for i in test])
    
    
    model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, 
              validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500,
              batch_size=512, show_metric=True, run_id=MODEL_NAME)


        
    #tensorboard --logdir=foo:D:\Python projects\Python plays gta v\log

model.save(MODEL_NAME)

#variable_summaries()