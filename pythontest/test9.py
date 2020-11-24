# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:08:28 2020

@author: admin
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD
import numpy as np

#y = lambda x: 2*x[0]*1.883 + 0.125*x[1]**2.0901 + 0.127*x[2]**0.363 + 3.14*x[3]**2.136 - 2.17*x[4]**2.17 + 6*x[5] - x[6] + 3*x[7]**2.1 + x[8]
#y = lambda x: 2*x[0] + 0.125*x[1] + 0.127*x[2] + 3.14*x[3] - 2.17*x[4] + 6*x[5] - x[6] + 3*x[7] + x[8]    

#y = lambda x: 2*x[0]*1.883 + 0.125*x[1]**2.0901 * 0.127*x[2]**0.363 ** 3.14*x[3]**2.136 - 2.17*x[4]**2.17 + 6*x[5] - x[6] * 3*x[7]**2.1 * x[8]
y = lambda x: 2*x[0] ** 0.125*x[1] + 0.127*x[2] * 3.14*x[3] * 2.17*x[4] * 6*x[5] * x[6] + 3*x[7] - 2.632*x[8]    
'''
trainX1 = np.random.randint(0,6,9)
#trainX2 = np.random.randint(0,6,9)
#trainX3 = np.random.randint(0,6,9)
#trainX4 = np.random.randint(0,6,9)
#trainX5 = np.random.randint(0,6,9)
#trainX6 = np.random.randint(0,6,9)


#trianY1 = y(list(trainX1))
#trianY2 = y(list(trainX2))
#trianY3 = y(list(trainX3))
#trianY4 = y(list(trainX4))
#trianY5 = y(list(trainX5))
#trianY6 = y(list(trainX6))


#x1 = np.array([trainX1,trainX2,trainX3,trainX4,trainX5,trainX6])
#y1 = np.array([trianY1,trianY2,trianY3,trianY4,trianY5,trianY6])

trainY1 = np.array(y(trainX1)).reshape(1,)
trainX1 = trainX1.reshape(-1,9)

'''
X = []
Y = []

for i in range(100000):
    trainX1 = np.random.randint(1,6,9)
    trainY1 = np.array(y(trainX1))
    X.append(trainX1)
    Y.append(trainY1)
    
X = np.array(X)
Y = np.array(Y)    



model = Sequential()
model.add(Dense(1028,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(1))


model.compile(optimizer=Adam(lr=0.005),metrics=['acc'],loss='mse')
model.fit(X,Y,batch_size=1280,epochs=3000)



testX = np.random.randint(1,6,9)
testX1 = testX.reshape(1,9)

model.predict(testX1),y(list(testX))


res = []
for i in range(10000):
    testX = np.random.randint(1,6,9)
    testX1 = testX.reshape(1,9)

    if abs(model.predict(testX1)-y(list(testX))) > 10:
        res.append([testX,model.predict(testX1),y(list(testX))])  
import pandas as pd        
res = pd.DataFrame(res)


model.predict(np.array(X[0]).reshape(1,9)),y(X[0])



import tensorflow as tf
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(model.weights[0]))
    print(sess.run(model.weights[1]))



