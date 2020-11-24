# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 20:59:06 2020

@author: admin
"""
from keras import models, layers, applications
model = applications.Xception(include_top=False, weights=None, 
                                    input_shape=(90, 90, 3), pooling='avg')
from keras.utils import plot_model
plot_model(model,to_file="model.png",show_shapes=True)
