# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:40:49 2020

@author: admin
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD
from keras.utils.np_utils import to_categorical

(train_img,train_label),(test_img,test_label) = mnist.load_data()


#数据预处理
train_img = train_img.reshape(-1,28*28)
train_img = train_img / 255

test_img = test_img.reshape(-1,28*28)
test_img = test_img / 255

train_label = to_categorical(train_label,num_classes =10)
test_label = to_categorical(test_label,num_classes =10)


#设置超参数
lr = 0.5
bath_size = 128
epochs = 10

model = Sequential()
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer=SGD(lr=lr),loss='categorical_crossentropy',metrics=['acc'])
model.fit(train_img,train_label,batch_size=bath_size,epochs=epochs)


tuple(model.evaluate(test_img,test_label))
model.predict(test_img[0].reshape(1,-1))