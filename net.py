from dataloader import loadImageSet,loadLabelSet
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv_layer=nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5),
                nn.MaxPool2d(kernel_size=2),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5),
                nn.MaxPool2d(kernel_size=2),
            )
        )
        self.fully_connected=nn.Sequential(
            nn.Linear(4*4*64,1024),
            nn.Sigmoid(),
            nn.Linear(1024,10),
            nn.Softmax()
        )

    def forward(self,x):
        out=self.conv_layer(x)
        out=out.view(out.size(0),-1)
        out=self.fully_connected(out)
        return out