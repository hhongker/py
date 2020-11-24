# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:39:45 2020

@author: admin
"""

import torch
import torch.nn as nn


X = torch.randn(3,4)

linear = nn.Linear(4,2)

Y = torch.randn(3,2)




optim = torch.optim.SGD(linear.parameters(),lr=0.1)
criterion=nn.MSELoss()


for step in range(1000):
     out = linear(X)
     
     loss = criterion(out, Y)
     
     optim.zero_grad()
     
     loss.backward()
     
     optim.step()
     
     print('loss:{}'.format(loss))
     
     
# In[]


import torch 
import torch.nn as nn
import torchvision.transforms as transforms

import torchvision




#超参数
lr = 0.1
batch_size = 128
epochs = 5




train_dataset = torchvision.datasets.MNIST('datasets', download=True,transform=transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)



test_dataset = torchvision.datasets.MNIST('datasets',train=False,transform=transforms.ToTensor())
test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)




model = nn.Sequential(
        nn.Linear(784,100),
        nn.ReLU(),
        nn.Linear(100,10)
        )

criterion = nn.CrossEntropyLoss()

optim = torch.optim.Adam(model.parameters(),lr=lr)

for epoch in range(epochs):
    for i,(images,labels) in enumerate(train_dataloader):
        
        images = images.reshape(-1,784)
        out = model(images)
        loss = criterion(out,labels)
        
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
    print('loss:',loss.item())
        

with torch.no_grad():
    correct = 0
    total = 0
    for i, (images,labels) in enumerate(test_dataloader):
        
        imgaes = images.reshape(-1,784)
        out = model(imgaes)
        correct += (torch.argmax(out,1)==labels).sum().item()
        total += images.shape[0]
        
    print('acc',correct / total)


# In[] 
import torch 
import torch.nn as nn
import torchvision.transforms as transforms

import torchvision


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#超参数
lr = 0.1
batch_size = 128
epochs = 5


train_dataset = torchvision.datasets.MNIST('datasets', download=True,transform=transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)



test_dataset = torchvision.datasets.MNIST('datasets',train=False,transform=transforms.ToTensor())
test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        
        self.layer1 = nn.Sequential(
                nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2))


        self.layer2 = nn.Sequential(
                nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2))
        

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(7*7*32,100)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.sofmax = nn.Linear(100,10)
        
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.relu3(out)
        out = self.dropout(out)
        
        out = self.sofmax(out)
        return out



model = NeuralNetwork().to(device)


criterion = nn.CrossEntropyLoss()


optim = torch.optim.Adam(model.parameters())


for epoch in range(epochs):
    for i,(images,labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        out = model(images)
        loss = criterion(out,labels)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
    print('loss:',loss.item())


model.eval()
#mode.train()


with torch.no_grad():
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(test_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        out = model(images)
        correct += (torch.argmax(out,1)==labels).sum().item()
        total += images.shape[0]
        
    print('acc',correct / total)




