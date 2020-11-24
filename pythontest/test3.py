# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 20:02:24 2020

@author: admin
"""

from keras.datasets import mnist
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np

(trainX,trainY),(testX,testY) = mnist.load_data()

#归一化
trainX = trainX / 255.0
testX = testX / 255.0

trainY = to_categorical(trainY,10)
testY = to_categorical(testY,10)


#trainX = trainX.reshape(-1,784)
#testX = testX.reshape(-1,784)
trainX = np.expand_dims(trainX,-1)
testX = np.expand_dims(testX,-1)



model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100,activation='relu' ,input_shape=(784,)))
model.add(Dense(10,activation='softmax'))

model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['acc'])

model.fit(trainX,trainY,batch_size=128,epochs=10)


model.evaluate(testX,testY)

model.predict(testX[:2])