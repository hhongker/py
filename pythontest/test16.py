# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:58:49 2020

@author: admin
"""

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import time

#设置超参数
batch_size = 128
epochs = 5
lr = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_sampler = torch.utils.data.SubsetRandomSampler(list(range(45000)))
valid_sampler = torch.utils.data.SubsetRandomSampler(list(range(45000,50000)))

train_dataset = torchvision.datasets.CIFAR10('datasets',
                                             download=True,
                                             transform=transforms.ToTensor())

#train_dataloader=torch.utils.data.DataLoader(train_dataset,
#                                             batch_size=batch_size,
#                                             shuffle=True)

train_dataloader=torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             sampler=train_sampler)


valid_dataloader=torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             sampler=valid_sampler)



test_dataset = torchvision.datasets.CIFAR10('datasets',
                                             download=True,
                                             transform=transforms.ToTensor())

test_dataloader=torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size)



class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
#        self.layer1 = nn.Sequential(nn.Conv2d(3,32,3,stride=1,padding=1),
#                                    nn.ReLU(),
#                                    nn.MaxPool2d(2,2))
#        self.layer2 = nn.Sequential(nn.Conv2d(32,64,3,stride=1,padding=1),
#                                    nn.ReLU(),
#                                    nn.MaxPool2d(2,2))

        vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg_features = vgg.features
        
#        将vgg特征提取的那部分参数固定住，不去做修改
#        for  parameter in self.vgg_features.parameters():
#            parameter.requires_grad = False
        
        self.flatten = nn.Flatten()
        
        self.linear = nn.Linear(1*1*512,100)
        self.relu = nn.ReLU()
        self.softmax = nn.Linear(100,10)
        
        
    def forward(self,x):
#        out = self.layer1(x)
#        out = self.layer2(out)

        out = self.vgg_features(x)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.relu(out)
        out = self.softmax(out)
        
        return out

model = CNN().to(device)




criterion = nn.CrossEntropyLoss()

optim = torch.optim.Adam(model.parameters())



for epoch in range(epochs):
    start = time.time()
    for i, (images,labels) in enumerate(train_dataloader):
        
        images = images.to(device)
        labels = labels.to(device)
        
        out = model(images)
        loss = criterion(out, labels)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
    with torch.no_grad():
        correct = 0
        total = 0
        
        for i, (images,labels) in enumerate(valid_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            out = model(images)
            correct += (torch.argmax(out,1)==labels).sum().item()
            total += images.shape[0]

    print('valid acc:', correct/total)
        
    print('loss:{},time:{:.2f}'.format(loss,time.time()-start))    
        


with torch.no_grad():
    correct = 0
    total = 0
    
    for i, (images,labels) in enumerate(test_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        out = model(images)
        correct += (torch.argmax(out,1)==labels).sum().item()
        
        total += images.shape[0]
        
        
    print('test acc:', correct/total)