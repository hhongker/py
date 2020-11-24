#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 19:53:17 2020

@author: math308
"""
import tensorflow as tf
import os
from tensorflow.python.client import device_lib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "99"

if __name__ == "__main__":
    print(device_lib.list_local_devices())
    print(tf.test.is_built_with_cuda())
    
# In[]
import tensorflow as tf
print (tf.__version__)
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
    
    
# In[]
with tf.device('/gpu:0'):
    print(111)
    
# In[]
tf.test.is_gpu_available()
print(tf.test.is_gpu_available())
print(222)