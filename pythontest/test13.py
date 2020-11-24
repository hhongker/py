# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 20:21:19 2020

@author: admin
"""

#爬取数据






#人脸定位
import face_recognition
import cv2


def iddefine(img_path='./'):
    image = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(image)
    
    y0, x1, y1, x0 =face_locations[0]
    return x0,y0,x1,y1



#画图保存
def saveImg(x0,y0,x1,y1,path_name):
    img = cv2.imread(path_name)
    cv2.rectangle(img, (x0,y0), (x1,y1), (0,255,0), 2)
    cv2.imwrite(path_name[:-4]+'new'+path_name[-4:], img)



#展示图片
def exehibit(x0,y0,x1,y1,path_name):
    img = cv2.imread(path_name)
    cv2.rectangle(img, (x0,y0), (x1,y1), (0,255,0), 2)
    cv2.imshow('image',img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    
    




def detect(filename):
    # cv2级联分类器CascadeClassifier,xml文件为训练数据
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # 读取图片
    img = cv2.imread(filename)
    # 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 进行人脸检测
    faces = face_cascade.detectMultiScale(gray,1.1,1,cv2.CASCADE_SCALE_IMAGE,(40,40),(100,100))



    # 绘制人脸矩形框
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # 命名显示窗口
    cv2.namedWindow('people')
    # 显示图片
    cv2.imshow('people', img)
    # 保存图片
#    cv2.imwrite('cxks.png', img)
    # 设置显示时间,0表示一直显示
    cv2.waitKey(0)





img_path=r'd:\Desktop\pythontest\people\mydata\1.jpg'

x0,y0,x1,y1 = iddefine(img_path)

exehibit(x0,y0,x1,y1,img_path)


detect(img_path)







