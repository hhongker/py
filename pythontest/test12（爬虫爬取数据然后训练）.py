# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 21:32:45 2020

@author: admin
"""
# In[爬取数据 （哈士奇，花朵，飞机，美女）]
from bs4 import BeautifulSoup
from selenium import  webdriver
import requests
import time
import os


dog_url = 'https://cn.bing.com/images/search?q=%E5%93%88%E5%A3%AB%E5%A5%87&qpvt=%E5%93%88%E5%A3%AB%E5%A5%87&form=IGRE&first=1&scenario=ImageBasicHover'
huaduo_url = 'https://cn.bing.com/images/search?q=%e8%8a%b1%e6%9c%b5&qpvt=%e8%8a%b1%e6%9c%b5&form=IGRE&first=1&scenario=ImageBasicHover'
feiji_url = 'https://cn.bing.com/images/search?q=%e9%a3%9e%e6%9c%ba&qpvt=%e9%a3%9e%e6%9c%ba&form=IGRE&first=1&scenario=ImageBasicHover'
#yuer_url = 'https://cn.bing.com/images/search?q=%e9%b1%bc%e5%84%bf&qpvt=%e9%b1%bc%e5%84%bf&form=IGRE&first=1&scenario=ImageBasicHover'
meinv_url = 'https://cn.bing.com/images/search?q=%e7%be%8e%e5%a5%b3&qpvt=%e7%be%8e%e5%a5%b3&form=IGRE&first=1&scenario=ImageBasicHover'

urls = {'dog':dog_url, 'huaduo':huaduo_url, 'feiji':feiji_url, 'yuer':yuer_url, 'meinv':meinv_url}



def crawling(url='#', path='img', name = '哈士奇'):
    driver = webdriver.Chrome()
    driver.get(url)
    
    i = 1
    while True:
        driver.execute_script('window.scrollTo(100000,document.body.scrollHeight);')
        time.sleep(1)
        i += 1
        print(i)
        if i >= 35:
            break
    
    html = driver.page_source
    
    soup = BeautifulSoup(html, 'lxml')

    content = soup.select('#vm_c > div.dg_b li div > div > a > div > img')
    
    img_src = []
    for i in content:
       img_src.append(i['src'])
    
    path = './img'
    if not os.path.exists(path):
            os.mkdir(path)
          
    new_path = path+"/"+name
    if not os.path.exists(new_path):
            os.mkdir(new_path)
    j = 0        
    for i in img_src:
        with open(path+'/{}/{}{}'.format(name,j,name+'.png'),'wb') as fw:
            try:
                rpp = requests.get(i)
                fw.write(rpp.content)
            except:
                fw.write(i.encode('utf-8'))
        j += 1
    driver.close()
    
for i,j in urls.items():
    crawling(url=j, path=i,name=i)
# In[]
import cv2
import os

def changeImage(imgpath = 'd:/Desktop/pythontest/img'):
    if(os.path.isdir(imgpath)):
        for img_name in os.listdir(imgpath):
            filepath = imgpath+'/'+img_name
            changeImage(filepath)
    if os.path.isfile(imgpath):
        try:
            image = cv2.imread(imgpath)
            image  = cv2.resize(image,(128,128),interpolation=cv2.INTER_AREA)
            cv2.imwrite(imgpath,image)
        except:
            print(imgpath)
            os.remove(imgpath)
            print('这不是图片或文件夹')          

changeImage()

# In[]
import os
import random
import shutil	
def splitSet(fileDir='./img/', train_dir='./newimg/', test_dir='./newimg/'):
    

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.1
    number = int(filenumber * rate)           # 按照rate比例从文件夹中取数据
    sample = random.sample(pathDir, number)  # 随机选取picknumber数量的数据

    for name in sample:#测试集
        shutil.copy(fileDir + name, test_dir + name)

    for filename in os.listdir(fileDir):#训练集
        if filename not in os.listdir(test_dir):
            shutil.copy(os.path.join(fileDir,filename),os.path.join(train_dir,filename))
    return


base_dir = './img/'
new_dir = './newimg/'
for i in os.listdir(base_dir):
    splitSet(fileDir=base_dir+i+'/',train_dir=new_dir+'/train/'+i+'/',test_dir=new_dir+'/test/'+i+'/')



# In[]
import cv2
import os
import numpy as np

train_img = []
test_img = []


def saveImage(data=[],imgpath = 'd:/Desktop/pythontest/img'):
    if(os.path.isdir(imgpath)):
        for img_name in os.listdir(imgpath):
            filepath = imgpath+'/'+img_name
            saveImage(data,filepath)
    if os.path.isfile(imgpath):
        try:
            image = cv2.imread(imgpath)
            data.append(image)
        except:
            print('这不是图片或文件夹') 


saveImage(data=train_img, imgpath='d:/Desktop/pythontest/newimg/train')
saveImage(data=test_img, imgpath='d:/Desktop/pythontest/newimg/test')


train_img = np.array(train_img)
test_img = np.array(test_img)
train_labels = np.array([0]*452 + [1]*647 + [2]*419 + [3]*471)
test_labels = np.array([0]*50 + [1]*71 + [2]*46 + [3]*52)


np.savez('./newimg/define_data.npz', ((train_img, train_labels),(test_img, test_labels)))

(tri,trl),(tei,tel) = np.load('./newimg/define_data.npz',allow_pickle=True)['arr_0']

# In[搭建神经网络]
import numpy as np

from keras.layers import Dense,Conv2D, MaxPool2D, Flatten
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras.optimizers import Adam,SGD
#读取数据
(train_img,train_labels),(test_img,test_labels) = np.load('./newimg/define_data.npz',allow_pickle=True)['arr_0']

#数据预处理
train_img = train_img / 255
test_img = test_img / 255

train_labels = to_categorical(train_labels,num_classes=4)
test_labels = to_categorical(test_labels,num_classes=4)


#设置超参数
lr = 0.05
batch_size = 128
epochs = 100


#搭建网络
model = Sequential()
model.add(Conv2D(6,[5,5],padding='SAME',activation='relu'))
model.add(MaxPool2D([2,2],[2,2]))
model.add(Conv2D(3,[3,3],padding='SAME',activation='relu'))
model.add(MaxPool2D([2,2],[2,2]))
model.add(Conv2D(3,[1,1],padding='SAME',activation='relu'))
model.add(MaxPool2D([2,2],[2,2]))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(4,activation='softmax'))

model.compile(SGD(lr=lr),loss='categorical_crossentropy',metrics=['acc'])
model.fit(train_img,train_labels,batch_size=batch_size,epochs=epochs)



print('测试集loss和acc分别是：',model.evaluate(x=test_img, y=test_labels,batch_size=128))
print('训练集loss和acc分别是：',model.evaluate(x=train_img, y=train_labels,batch_size=128))

res_rate = 0
for i in range(len(test_img)):
    pre_res = np.argmax(model.predict(test_img[i:i+1])[0])
    real_res = np.argmax(test_labels[i])
    print(pre_res, real_res)
    if pre_res == real_res:
        res_rate += 1
    
print('正确率：{}'.format(res_rate/len(test_img)))
