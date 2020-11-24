# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 21:43:33 2020

@author: admin
"""

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical


(trainX,trainY),(testX,testY) =  cifar10.load_data()

trainX = trainX / 255
testX = testX / 255
trainY = to_categorical(trainY)
testY = to_categorical(testY)

model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=(32,32,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3),padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))


model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['acc'])
model.fit(trainX,trainY,batch_size=256,epochs=10)