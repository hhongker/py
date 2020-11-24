# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 21:43:16 2020

@author: admin
"""

import cv2
import matplotlib.pyplot as plt
 
def show(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()
 
def imread(image):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
 
def facedetect(image):
    image = imread(image)
    # 级联分类器
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=2, minSize=(2, 2),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
 
    for (x, y, w, h) in rects:
        # 画矩形框
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
    show(image)
    
    
facedetect('d:/Desktop/ceshi.jpg')    



import numpy as np  
import cv2  
  
  
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  
eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")  
  
img = cv2.imread('d:/Desktop/ceshi.jpg')  
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
                      
faces = face_cascade.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(50,50),(100,100))  
  
if len(faces)>0:  
    for faceRect in faces:  
        x,y,w,h = faceRect  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2,8,0)
        roi_gray = gray[y:y+h,x:x+w]  
        roi_color = img[y:y+h,x:x+w]  
  
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,1,cv2.CASCADE_SCALE_IMAGE,(2,2))  
        for (ex,ey,ew,eh) in eyes:  
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)  
              
cv2.imshow("img",img)  
cv2.waitKey(0)  