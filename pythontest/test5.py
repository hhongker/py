# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 16:52:48 2020

@author: admin
"""

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.applications import VGG16,ResNet50

(trainX,trainY),(testX,testY) =  cifar10.load_data()

trainX = trainX / 255
testX = testX / 255
trainY = to_categorical(trainY)
testY = to_categorical(testY)



vgg = VGG16(include_top=False,weights=None,input_shape=(32,32,3))
model_vgg = Sequential()

model_vgg.add(vgg)
model_vgg.add(Flatten())
model_vgg.add(Dense(100,activation='relu'))
model_vgg.add(Dense(10,activation='softmax'))


model_vgg.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['acc'])
model_vgg.fit(trainX,trainY,batch_size=256,epochs=10)





model_RN50 = Sequential()
rn50 = ResNet50(include_top=False,input_shape=(32,32,3))

model_RN50.add(rn50)
model_RN50.add(Flatten())
model_RN50.add(Dense(100,activation='relu'))
model_RN50.add(Dense(10,activation='softmax'))
model_RN50.layers[0].trainable = False

model_RN50.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['acc'])
model_RN50.fit(trainX,trainY,batch_size=256,epochs=10)
