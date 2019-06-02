import tensorflow as tf
import numpy as np

p = tf.placeholder(shape = [None,4],dtype=tf.float32)
t = tf.nn.softmax(p)
t1      = tf.random.categorical(tf.log(t),1)
t2 = tf.one_hot(t1, 4,
           on_value=1.0, off_value=0.0,
           axis=-1)


with tf.Session() as sess:
    inArray = [[0.8,0.5,0.1,0.2]]
    index, outArray = sess.run([t1,t2],feed_dict={p:inArray})
    print("Index:",index)
    print("Array:",outArray)
   

# =============================================================================
# def f1(): return tf.constant(17)
# def f2(): return tf.constant(23)
# def f3(): return tf.constant(-1)
# =============================================================================

def roll():  return tf.constant([0,0,0,0,0,0,0,0,0])
def rollR(): return tf.constant([0,1,0,0,0,0,0,0,0])
def rollL(): return tf.constant([0,0,1,0,0,0,0,0,0])
def acc():   return tf.constant([0,0,0,1,0,0,0,0,0])
def accR():  return tf.constant([0,0,0,0,1,0,0,0,0])
def accL():  return tf.constant([0,0,0,0,0,1,0,0,0])
def brakeR():return tf.constant([0,0,0,0,0,0,1,0,0])
def brakeL():return tf.constant([0,0,0,0,0,0,0,1,0])
def brake(): return tf.constant([0,0,0,0,0,0,0,0,1])

# =============================================================================
# while True:
#     time0 = time.time()
#     #-----------------zzx
#     if collectData:
#         state = get_state()
#         action_out, gradients = sess.run([myAgent.action_out,myAgent.gradients], feed_dict={myAgent.state_in:[state]})
# 
#         if do_action:
#             releaseKeys();
#             update_pressed_keys(categoriesToKeys(action))
#             
#         reward = reward + delta_time
#         current_rewards.append(reward)
#         current_gradients.append(gradients)
# =============================================================================
