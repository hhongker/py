# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 09:34:50 2020

@author: admin
"""
import numpy as np
import tensorflow as tf
import math
import random
#
#y = lambda x: 2*x[0] + 0.125*x[1] + 0.127*x[2] + 3.14*x[3] - 2.17*x[4] + 6*x[5] - x[6] + 3*x[7] + x[8]  
y = lambda x: 2*x[0]*1.883 + 0.125*x[1]**2.0901 + 0.127*x[2]**0.363 + 3.14*x[3]**2.136 - 2.17*x[4]**2.17 + 6*x[5] - x[6] + 3*x[7]**2.1 + x[8]  
    
'''
trainX1 = np.random.randint(0,6,9)
trainX2 = np.random.randint(0,6,9)
trainX3 = np.random.randint(0,6,9)
trainX4 = np.random.randint(0,6,9)
trainX5 = np.random.randint(0,6,9)
trainX6 = np.random.randint(0,6,9)


trianY1 = y(list(trainX1))
trianY2 = y(list(trainX2))
trianY3 = y(list(trainX3))
trianY4 = y(list(trainX4))
trianY5 = y(list(trainX5))
trianY6 = y(list(trainX6))


x1 = np.array([trainX1,trainX2,trainX3,trainX4,trainX5,trainX6])
y1 = np.array([trianY1,trianY2,trianY3,trianY4,trianY5,trianY6])
'''
X = []
Y = []

for i in range(10000):
    trainX1 = np.random.randint(0,6,9)
    trainY1 = np.array(y(trainX1))
    X.append(trainX1)
    Y.append(trainY1)
    
X = np.array(X)
Y = np.array(Y) 



V_X = []
V_Y = []

for i in range(1000):
    trainX1 = np.random.randint(0,6,9)
    trainY1 = np.array(y(trainX1))
    V_X.append(trainX1)
    V_Y.append(trainY1)
    
V_X = np.array(V_X)
V_Y = np.array(V_Y) 



W = {
        'w1':tf.Variable((tf.random_normal([9,18],stddev=0.1))),
        'w2':tf.Variable((tf.random_normal([18,9],stddev=0.1))),
        'w3':tf.Variable((tf.random_normal([9,6],stddev=0.1))),
        'w4':tf.Variable((tf.random_normal([6,6],stddev=0.1))),
        'w5':tf.Variable((tf.random_normal([6,1],stddev=0.1)))                
        }


B = {
        'b1':tf.Variable((tf.random_normal([18],stddev=0.1))),
        'b2':tf.Variable((tf.random_normal([9],stddev=0.1))),
        'b3':tf.Variable((tf.random_normal([6],stddev=0.1))),
        'b4':tf.Variable((tf.random_normal([6],stddev=0.1))),
        'b5':tf.Variable((tf.random_normal([1],stddev=0.1)))
        }


#超参数
lr = 0.0001
ecoph = 20000
bath_size = 2048



input_X = tf.placeholder(tf.float32,[None,9])
output_Y = tf.placeholder(tf.float32,[None,1])



w1 = tf.matmul(input_X,W['w1'])
B1 = tf.nn.sigmoid(w1 + B['b1'])

w2 = tf.matmul(B1,W['w2'])
B2 = tf.nn.relu(w2 + B['b2'])

w3 = tf.matmul(B2,W['w3'])
B3 = tf.nn.relu(w3 + B['b3'])

w4 = tf.matmul(B3,W['w4'])
B4 = tf.nn.sigmoid(w4 + B['b4'])

w5 = tf.matmul(B4,W['w5'])
output = w5 + B['b5']




#cost=tf.losses.mean_squared_error(output_Y, output)
cost=tf.reduce_mean(tf.square(output-output_Y))
optimizer=tf.train.GradientDescentOptimizer(lr).minimize(cost)


init=tf.global_variables_initializer()

#with tf.Session() as sess:
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 训练循环
for epoch in range(ecoph):
    total_batch = math.ceil(len(X) / bath_size)

    # 遍历所有 batch
    for i in range(total_batch):
#            index = np.random.randint(0,10000,bath_size)
        index = random.sample(range(10000),bath_size)

        batch_X, batch_Y = X[index],Y[index]#X[i*bath_size:],Y[i*bath_size:]
        sess.run(
            optimizer,
            feed_dict={input_X: batch_X, output_Y: np.array(batch_Y).reshape(-1,1)})

    # 每循环10次  打印一次状态
    if epoch % 10 == 0:
        val_loss = sess.run(
            cost,
            feed_dict={
                input_X:V_X,
                output_Y:V_Y.reshape(-1,1)})
        print('Epoch {:<3} - val_loss: {}'.format(
            epoch,
            val_loss))
    






testX = np.random.randint(0,6,9)
testX1 = testX.reshape(1,9)

#sess.run(tf.global_variables_initializer())
pre_result = sess.run(
            output,
            feed_dict={input_X:testX1})
result = y(testX)
print('predict_val: {},val:{}'.format(pre_result,result))



sess.close()








