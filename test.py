import torch
from dataloader import loadLabelSet,loadImageSet

x_test=loadImageSet("mnist_dataset/t10k-images-idx3-ubyte")
y_test=loadLabelSet("mnist_dataset/t10k-labels-idx1-ubyte")
x_test=torch.tensor(x_test).float()

net=torch.load("data/model.pt")

pre=net(x_test)

def find_idx(pre):
    idx=[]
    for i in range(len(pre)):
        max=0
        for j in range(len(pre[i])):
            if pre[i][j]>=max:
                max=pre[i][j]
                index=j
        idx.append(index)
    return idx

def acc(pre,real):
    right=0
    for i in range(len(pre)):
        if pre[i]== real[i]:
            right+=1
    return right/len(pre)

predict=find_idx(pre)
print("accuracy:",acc(y_test,predict)*100,"%")
