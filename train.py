from dataloader import loadImageSet,loadLabelSet
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from net import Net

images=loadImageSet("mnist_dataset/train-images-idx3-ubyte")
labels=loadLabelSet("mnist_dataset/train-labels-idx1-ubyte")
'''for i in range(len(images)):
    for j in range(len(images[i])):
        if(images[i][j])==0 :
            images[i][j]=0.01

imgs=[]
tran=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
    ])

for img in images:
    img=img.reshape(28,28)
    print(img.shape)
    img=tran(img)
    imgs.append(img)'''

I=np.identity(10)
labels=I[labels]
labels=torch.tensor(labels).float()
for i in range(len(labels)):
    for j in range(len(labels[i])):
        if labels[i][j]==1:
            labels[i][j]=0.99
        else :
            labels[i][j]=0.01


images=torch.tensor(images).float()
print("datasets loaded")

#net=Net()
net=nn.Sequential(
    nn.Linear(784,256),
    nn.Sigmoid(),
    nn.Linear(256,64),
    nn.Sigmoid(),
    nn.Linear(64,10),
    nn.Softmax()
)
optim=torch.optim.Adam(net.parameters(),lr=0.01)
loss_F=nn.MSELoss()
'''for epoch in range(1):
    for i in range(len(imgs)):
        img=imgs[i].unsqueeze(0).float()
        label=labels[i]
        out=net(img)
        loss=loss_F(out,label)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i %100 ==0:
            print("loss",loss.item())'''
for epoch in range(1000):
    out=net(images)
    loss=loss_F(out,labels)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print("loss:",loss.item())


torch.save(net,"data/model.pt")

