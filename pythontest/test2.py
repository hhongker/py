# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 19:28:46 2020

@author: Dr. Tang
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('./datasets',one_hot=True,reshape=False)

#三种数
#超参数
learning_rate=0.01
batch_size=128
epochs=10


tf.reset_default_graph()
#数据
inputs=tf.placeholder(tf.float32,[None,28,28,1])
labels=tf.placeholder(tf.float32,[None,10])
##Tensorboard使用一般分三步
#Tensorboard使用一般分三步第一步，实例化一个writer
#变量
writer=tf.summary.FileWriter('logs')

#Tensorboard第二步，检测需要记录的数据
#weights={'wc1':tf.Variable(tf.random_normal([5,5,1,6],stddev=0.1)),
#         'wc2':tf.Variable(tf.random_normal([5,5,6,16],stddev=0.1)),
#         'fc1':tf.Variable(tf.random_normal([5*5*16,120],stddev=0.1)),
#         'fc2':tf.Variable(tf.random_normal([120,84],stddev=0.1)),
#         'out':tf.Variable(tf.random_normal([84,10],stddev=0.1))}
#
#biases={'bc1':tf.Variable(tf.zeros([6])),
#        'bc2':tf.Variable(tf.zeros([16])),
#        'bf1':tf.Variable(tf.zeros([120])),
#        'bf2':tf.Variable(tf.zeros([84])),
#        'bout':tf.Variable(tf.zeros([10]))}

#weights_conv1=tf.Variable(tf.random_normal([5,5,1,6],stddev=0.1))
#weights_conv2=tf.Variable(tf.random_normal([5,5,6,16],stddev=0.1))
#weights_fc1=tf.Variable(tf.random_normal([5*5*16,120],stddev=0.1))
#weights_fc2=tf.Variable(tf.random_normal([120,84],stddev=0.1))
#weights_out=tf.Variable(tf.random_normal([84,10],stddev=0.1))
#
#bias_conv1=tf.Variable(tf.zeros([6]))
#bias_conv2=tf.Variable(tf.zeros([16]))
#bias_fc1=tf.Variable(tf.zeros([120]))
#bias_fc2=tf.Variable(tf.zeros([84]))
#bias_out=tf.Variable(tf.zeros([10]))

#搭网络
#conv1=tf.nn.conv2d(inputs,weights['wc1'],[1,1,1,1],padding='SAME')
#conv1=tf.nn.relu(conv1+biases['bc1'])
##tf.summary.histogram('conv1',conv1)
#
#pool1=tf.nn.max_pool(conv1,[1,2,2,1],[1,2,2,1],padding='SAME')
#conv2=tf.nn.conv2d(pool1,weights['wc2'],[1,1,1,1],padding='VALID')
#conv2=tf.nn.relu(conv2+biases['bc2'])
##tf.summary.histogram('conv2',conv2)
#
#pool2=tf.nn.max_pool(conv2,[1,2,2,1],[1,2,2,1],padding='SAME')
#flatten1=tf.reshape(pool2,[-1,pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])
#
#fc1=tf.matmul(flatten1,weights['fc1'])+biases['bf1']
#fc1=tf.nn.relu(fc1)
#fc2=tf.matmul(fc1,weights['fc2'])+biases['bf2']
#fc2=tf.nn.relu(fc2)
#logits=tf.matmul(fc2,weights['out'])+biases['bout']
#tf.summary.histogram('logits',logits)


conv1 = tf.layers.conv2d(
        inputs,6,(5,5),padding = 'SAME',
        activation = tf.nn.relu,
        kernel_initializer = tf.random_normal_initializer()
                         )
pool1 = tf.layers.max_pooling2d(conv1,(2,2),(2,2),padding='SAME')

conv2 = tf.layers.conv2d(
        pool1,16,(5,5),padding = 'SAME',
        activation = tf.nn.relu,
        kernel_initializer = tf.random_normal_initializer()
                         )
pool2 = tf.layers.max_pooling2d(conv2,(2,2),(2,2),padding='SAME')

flatten = tf.layers.flatten(pool2)
fc1 = tf.layers.dense(
        flatten,
        120,
        activation=tf.nn.relu
        )
fc2 = tf.layers.dense(
        fc1,
        84,
        activation=tf.nn.relu
        )
logits = tf.layers.dense(
        fc2,
        10
        )




#算损失
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                    (logits=logits,labels=labels))
tf.summary.scalar('cost',cost)
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
corrected_prediction=tf.equal(tf.arg_max(logits,1),tf.arg_max(labels,1))
accuracy=tf.reduce_mean(tf.cast(corrected_prediction,tf.float32))
merged=tf.summary.merge_all()
init=tf.global_variables_initializer()

#运行网络
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    sess.run(init)
    #Tensorboard第三步，在sess里面添加graph
    writer.add_graph(sess.graph)
    num_batches=mnist.train.num_examples//batch_size+1
    for epoch in range(epochs):
        for batch in range(num_batches):
            batches=mnist.train.next_batch(batch_size)
            #Tensorboard第四步，把每次的结果存起来
            _,summary=sess.run([optimizer,merged],feed_dict={inputs:batches[0],
                                          labels:batches[1]})
            writer.add_summary(summary,epoch*num_batches+batch)
        acc=sess.run(accuracy,feed_dict={inputs:mnist.validation.images,
                                         labels:mnist.validation.labels})
        print('Epoch:{:<2},acc:{:.3f}'.format(epoch+1,acc))
    test_acc=sess.run(accuracy,feed_dict={inputs:mnist.test.images,
                                         labels:mnist.test.labels})
    print('test acc',test_acc)
        