# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 22:26:20 2020

@author: admin
"""
import torch
import torchvision
import torchvision.transforms as transforms
import cv2





with open('d:/Desktop/pythontest/datasets/imageNet.txt','r',encoding='utf-8') as f:
    data = f.readlines()
    
    
    
with open('d:/Desktop/pythontest/datasets/imageNet.txt','w',encoding='utf-8') as f:
    for i in data[::2]:
        f.write(i)
        
        


def readImage(path=r'd:/Desktop/pythontest/datasets/myimage/elephent1.jpg'):
    img = cv2.imread(path)
    
#    cv2.imshow('elepthent',img)
#    if cv2.waitKey(0) == ord('q'):
#        cv2.destroyAllWindow()

    img = cv2.resize(img,(224,224))
    
    return img
    






def preKind(img):
    vgg16 = torchvision.models.vgg16(pretrained=True)
    
    return torch.argmax(vgg16(transforms.ToTensor()(img).reshape([1,3,224,224]))).item()



img = readImage()
nums = preKind(img)
data[nums]    
    
    
