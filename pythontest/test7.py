# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 08:42:54 2020

@author: admin
"""
import keras
import numpy as np
import matplotlib.pyplot as plt

from keras import layers
import pandas as pd

x1 = np.random.randint(0,100,33)
x2 = x1 * 2 / 4 + np.random.randn()*8
                                 
x = pd.DataFrame({
    'x1':x1,
    'x2':x2
})

y = x1 ** 2 - x2*3 + np.random.rand()*7
                                   
                                   
model = keras.Sequential()

model.add(layers.Dense(1,input_dim=2))

model.summary()
model.compile(
    optimizer='adam',
    loss='mse'
)
model.fit(x,y,epochs=100)