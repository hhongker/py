# -*- coding: UTF-8 -*- 
#-------------------------------------------
# 用预训练好的ResNet50网络进行图像分类，并根据自己的数据微调网络权重
#-------------------------------------------
from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.layers import Flatten, Dense, Input, GlobalMaxPooling2D
from keras.models import Model
from keras.datasets import cifar10
from keras import utils
from keras.regularizers import l2
import numpy as np
import cv2
# In[1]: 加载数据

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
num_class = len(np.unique(y_train)) # 类别数

# 数据预处理
y_train=utils.to_categorical(y_train,num_classes=10)
y_test=utils.to_categorical(y_test,num_classes=10)
x_train = preprocess_input(x_train) # 规范化每张图片的像素值为 -1 到 1
x_test = preprocess_input(x_test) 

# 缩放原始图像到(im_w,im_h,3)
im_w=32
im_h=32
x_train_reshape = np.zeros((len(x_train),im_w,im_h,3))
for i in range(len(x_train)):
   img = x_train[i]
   img = cv2.resize(img,(im_w,im_h))
   x_train_reshape[i,:,:,:] = img

x_test_reshape = np.zeros((len(x_test),im_w,im_h,3))
for i in range(len(x_test)):
   img = x_test[i]
   img = cv2.resize(img,(im_w,im_h))
   x_test_reshape[i,:,:,:] = img
                
# In[2]: ResNet50模型，加载预训练权重
# 若没有模型文件，则自动下载（由于下载速度很慢，所以建议先把文件放进相应的目录）
# C:\Users\Administrator\.keras\models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
base_model = ResNet50(input_shape=(im_w, im_h, 3), 
                      include_top=False, 
                      weights='imagenet'#,pooling='avg'
                      )
base_model.trainable=False
x = base_model.output
x = Flatten()(x)
x = Dense(512,activation='relu',kernel_regularizer=l2(0.0003))(x)
predictions = Dense(num_class, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.summary() # 打印模型各层的输出大小
 
# In[3]: 训练
# 更改迭代此处 epochs=50，才能获得比较高的识别率
model.fit(x_train_reshape, y_train, epochs=50, batch_size=64,validation_split=0.1) 
#model.save_weights('resnet_turn_cifar10.h5') 

# In[4]:测试
loss, accuracy = model.evaluate(x_test_reshape, y_test)
print('测试识别率为：', np.round(accuracy,4))
