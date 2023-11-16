import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


from eunet import Eunet

class chain_eunet_4(nn.Module):
    def __init__(self):
        super(chain_eunet_4, self).__init__()
        self.eunet1 = Eunet()
        self.eunet2 = Eunet()
        self.eunet3 = Eunet()
        self.eunet4 = Eunet()
        self.conv01 = nn.Conv2d(in_channels=4,out_channels=3,kernel_size=3,padding=1)
        self.conv12 = nn.Conv2d(in_channels=5,out_channels=3,kernel_size=3,padding=1)
        self.conv23 = nn.Conv2d(in_channels=6,out_channels=3,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(3)
        self.bn3 = nn.BatchNorm2d(3)
    def forward(self,x):
        output1 = self.eunet1(x)
        input2 = self.bn1(self.conv01(torch.cat((x,output1),dim=1)))
        output2 = self.eunet2(input2)
        input3 = self.bn2(self.conv12(torch.cat((x,output1,output2),dim=1)))
        output3 = self.eunet3(input3)
        input4 = self.bn3(self.conv23(torch.cat((x,output1,output2,output3),dim=1)))
        output4 = self.eunet4(input4)
        return output4

if __name__ == '__main__':
    a = torch.rand(8,3,128,128)
    net = chain_eunet_4()
    b = net(a)
    print(b.shape)
