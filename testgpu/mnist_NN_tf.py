from tensorflow.keras import datasets, utils, Sequential, optimizers 
from tensorflow.keras import Input,layers

# In[数据加载]:
(x_train,y_train),(x_test,y_test)=datasets.mnist.load_data('D:/Desktop/test/mnist.npz')

# In[数据预处理]:
# 把像素值转换为 0到1
x_train=x_train/255.0
x_test=x_test/255.0

# 模型要求把标签转换为one hot格式，例如：2转换为[0,1,0,0,0,0,0,0,0,0]
y_train=utils.to_categorical(y_train,num_classes=10)
y_test=utils.to_categorical(y_test,num_classes=10)

# In[构建网络]:
# 输入层有28x28的图像，并转换为784维，中间层512维，输出层10维(对应10个类别)
model = Sequential()
model.add(Input(shape=(28,28)))
model.add(layers.Reshape((28*28,)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# In[训练]:
# 优化器为随机梯度下降SGD，损失函数为交叉熵损失categorical_crossentropy，性能评估用识别率accuracy
optimizer = optimizers.SGD(lr=0.5)
model.compile(optimizer,loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=128, epochs=5)

# 评估模型
print('\nTesting ------------------- ')
loss, accuracy = model.evaluate(x_test, y_test)
