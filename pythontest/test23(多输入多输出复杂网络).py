# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 13:39:08 2020

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from tensorflow.keras import datasets, utils, Sequential, optimizers,activations 
from tensorflow.keras import Input,layers,Model
import tensorflow as tf

# In[数据加载]:
(x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()

# In[数据预处理]:
# 把像素值转换为 0到1
x_train=x_train/255.0
x_test=x_test/255.0

# 模型要求把标签转换为one hot格式，例如：2转换为[0,1,0,0,0,0,0,0,0,0]
y_train=utils.to_categorical(y_train,num_classes=10)
y_test=utils.to_categorical(y_test,num_classes=10)

# In[构建网络]:
# 输入层有28x28的图像，并转换为784维，中间层512维，输出层10维(对应10个类别)
def base_model():
    inp = Input(shape=(32,32,3))
    
    x = layers.Conv2D(8, (5,5),activation='relu', padding='SAME')(inp)
    x = layers.MaxPooling2D((2,2),(2,2))(x)
    a1 = layers.Conv2D(6, (5,5),activation='relu', padding='SAME')(x)
    a2 = layers.Conv2D(6, (3,3),activation='relu', padding='SAME')(x)
    a3 = layers.Conv2D(6, (1,1),activation='relu', padding='SAME')(x)
    x = layers.add([a1, a2, a3])
    x = layers.MaxPooling2D((2,2),(2,2))(x)
    
    b1 = layers.Conv2D(4, (3,3),activation='relu', padding='VALID')(x)
    b1 = layers.Conv2D(8, (1,1),activation='relu', padding='SAME')(b1)
    
    b2 = layers.Conv2D(6, (5,5),activation='relu', padding='SAME')(x)
    b2 = layers.Conv2D(4, (3,3),activation='relu', padding='VALID')(b2)
    
    c1 = layers.Conv2D(6, (3,3),activation='relu', padding='VALID')(x)
    
    x = layers.concatenate([b1,b2,c1], axis=-1)
    
    f1 = layers.Flatten()(x)
    f1 = layers.Dense(256,activation='relu')(f1)
    model = Model(inp, f1)
    return model
# In[]
baseModel = base_model()
inp1 = baseModel.input
output = baseModel.output
o1 = layers.Dense(10,activation='softmax',name = 'o1')(output)
o2 = layers.Dense(10,activation='sigmoid',name = 'o2')(output)

model = Model(inp1,[o1,o2])



model.compile(optimizer= optimizers.SGD(lr=0.01),
              loss={'o1': 'categorical_crossentropy',
 			  'o2': 'mse',
              },
 			  loss_weights={'o1': 1,
 							'o2': .1,},
               metrics=['accuracy']
               )

# model.compile(optimizer= optimizers.SGD(lr=0.01),
#               loss=[ 'categorical_crossentropy',
# 			  'mse',
#               ],
# 			  loss_weights=[ 1,
# 							 .1,],
#               metrics=['accuracy']
#               )

model.fit(x_train, {'o1':y_train,
				   'o2':y_train,},
          validation_split=0.1,
          epochs=50,
          batch_size=64,
          )


# In[]

baseModel = base_model()


input1 = baseModel.input
inp1 = Input(shape=(32,32,3))
inp2 = Input(shape=(32,32,3))

# output = baseModel.output
o1 = baseModel(inp1)
o1 = layers.Dense(10,activation='softmax',name = 'o1')(o1)
o2 = baseModel(inp2)
o2 = layers.Dense(10,activation='sigmoid',name = 'o2')(o2)
model = Model([inp1,inp2],[o1,o2])

model.compile(optimizer= optimizers.SGD(lr=0.01),
              loss={'o1': 'categorical_crossentropy',
 			  'o2': 'mse',
              },
 			  loss_weights={'o1': 1,
 							'o2': .1,},
               metrics={'o1':'accuracy','o2':'accuracy'}
               )

model.fit([x_train,x_train], {'o1':y_train,
				   'o2':y_train,},
          validation_split=0.1,
          epochs=50,
          batch_size=64,
          )

# In[]
# 评估模型
print('\nTesting ------------------- ')
loss, accuracy = model.evaluate(x_test, y_test)