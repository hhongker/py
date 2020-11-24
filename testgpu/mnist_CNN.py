from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam

#载入数据
(x_train,y_train),(x_test,y_test)=mnist.load_data('/home/s10/test1/test/mnist.npz')

#（60000,28,28，）->(60000,28,28,1) 归一化
x_train=x_train.reshape(-1,28,28,1)/255.0
x_test=x_test.reshape(-1,28,28,1)/255.0

#换one hot格式
y_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10)

# 定义顺序模型
model = Sequential()

#第一个卷积层
#input_shape输入平面
#filter卷积核/滤波器个数
#kernel_size卷积窗口大小
#strides步长
#padding：padding 方式same/valid
#activation激活函数
# model.add(Convolution2D(
#     input_shape=(28,28,1),
#     nb_filter=32, #32个卷积核
#     kernel_size=5,  #卷积核5*5
#     strides=1,  #步长为1
#     padding='same',
#     activation='relu'
# ))
model.add(Convolution2D(
    input_shape=(28,28,1),
    nb_filter=32,#32个卷积核
    nb_col=5,#卷积核的列数为5
    nb_row=5,#卷积核的行数为5
    subsample=1, #步长为1
    border_mode='same',#边界模式
    activation='relu'
))

#第一个池化层
# model.add(MaxPooling2D(
#     pool_size=2,
#     strides=2,
#     padding='same'
# ))
model.add(MaxPooling2D(
    pool_size=(2,2),
    strides=2,
    border_mode='same'
))

#第二个卷积层
#model.add(Convolution2D(64,5,strides=1,padding='same',activation='relu'))
model.add(Convolution2D(64,5,subsample=1,border_mode='same',activation='relu'))

#第二个池化层
#model.add(MaxPooling2D(2,2,'same'))
model.add(MaxPooling2D((2,2),2,'same'))

#把第二个池化层的输出扁平化为1维
model.add(Flatten())

#第一个全连接层
model.add(Dense(1024,activation='relu'))
#Dropout
model.add(Dropout(0.5))
#第二个全连接层
model.add(Dense(10,activation='softmax'))

# 定义优化器
adam =Adam(lr=1e-4)

#定义优化器，loss,funtion训练过程中计算准确率
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
# model.fit(x_train,y_train,batch_size=32,epochs=10)
model.fit(x_train, y_train, batch_size=64, nb_epoch=10)

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss:', loss)
print('accuracy:', accuracy)