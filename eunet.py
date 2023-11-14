import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

class mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x
mish = mish()

class Conv_transition(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels):
        
        super(Conv_transition, self).__init__()
        if not kernel_size:
            kernel_size = [1, 3, 5]
        paddings = [int(a / 2) for a in kernel_size]
        
        self.Conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[0], stride=1, padding=paddings[0])
        self.Conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[1], stride=1, padding=paddings[1])
        self.Conv3 = nn.Conv2d(in_channels, out_channels, kernel_size[2], stride=1, padding=paddings[2])
        
        self.Conv_f = nn.Conv2d(3 * out_channels, out_channels, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        
        x1 = self.act(self.Conv1(x))
        x2 = self.act(self.Conv2(x))
        x3 = self.act(self.Conv3(x))
        
        x = torch.cat([x1, x2, x3], dim=1)
        return self.act(self.bn(self.Conv_f(x)))


class Eunet(nn.Module):
    def __init__(self):
        super(Eunet, self).__init__()

        self.incep1_2 = self.transition(3, 2)
        self.incep1_3 = self.transition(3, 3)
        self.incep1_4 = self.transition(3, 4)
        self.incep1_5 = self.transition(3, 5)
        self.incep1_6 = self.transition(3, 6)


        self.en2_1 = nn.Conv2d(20, 20, 1, padding=0)
        self.en2_1_bn = nn.BatchNorm2d(20)
        self.en2_2 = nn.Conv2d(20, 10, 3, stride=1, padding=1)
        self.en2_2_bn = nn.BatchNorm2d(10)
        self.en2_3 = nn.Conv2d(10, 5, 3, stride=1, padding=1)
        self.en2_3_bn = nn.BatchNorm2d(5)


        self.ad_max_pool_1 = nn.AdaptiveAvgPool2d(output_size=(int(8), int(8)))
        self.ad_max_pool_2 = nn.AdaptiveAvgPool2d(output_size=(int(16), int(16)))
        self.ad_max_pool_3 = nn.AdaptiveAvgPool2d(output_size=(int(32), int(32)))
        self.ad_max_pool_4 = nn.AdaptiveAvgPool2d(output_size=(int(64), int(64)))
        self.ad_max_pool_5 = nn.AdaptiveAvgPool2d(output_size=(int(128), int(128)))


        self.de2_1 = nn.Conv2d(35, 8, 3, stride=1, padding=1)
        self.de2_1_bn = nn.BatchNorm2d(8)
        self.de2_2 = nn.Conv2d(38, 16, 3, stride=1, padding=1)
        self.de2_2_bn = nn.BatchNorm2d(16)
        self.de2_3 = nn.Conv2d(46, 32, 3, stride=1, padding=1)
        self.de2_3_bn = nn.BatchNorm2d(32)
        self.de2_4 = nn.Conv2d(62, 2, 3, stride=1, padding=1)
        self.de2_4_bn = nn.BatchNorm2d(2)
        self.finalconv = nn.Conv2d(2,1,3, stride=1, padding=1)
        self.finalbn = nn.BatchNorm2d(1)



        self.soft1 = nn.Softmax(dim=1)



    def forward(self, x):
        incep1_2 = self.incep1_2(x)
        incep1_3 = self.incep1_3(x)
        incep1_4 = self.incep1_4(x)
        incep1_5 = self.incep1_5(x)
        incep1_6 = self.incep1_6(x)

        feature1_cat_1 = torch.cat((incep1_2, incep1_3, incep1_4, incep1_5, incep1_6), dim=1)
        
        encoder2_1 = F.max_pool2d(mish(self.en2_1_bn(self.en2_1(feature1_cat_1))), 2, 2)
        


        encoder2_2 = F.max_pool2d(mish(self.en2_2_bn(self.en2_2(encoder2_1))), 2, 2)
        encoder2_3 = F.max_pool2d(mish(self.en2_3_bn(self.en2_3(encoder2_2))), 2, 2)

        feature1_cat_2 = torch.cat((self.ad_max_pool_2(encoder2_1), self.ad_max_pool_2(encoder2_2), encoder2_3), dim=1)

        decoder2_1 = F.interpolate(mish(self.de2_1_bn(self.de2_1(feature1_cat_2))), scale_factor=(2, 2),
                                   mode='bilinear')

        feature1_cat_3 = torch.cat((self.ad_max_pool_3(encoder2_1), self.ad_max_pool_3(encoder2_2), decoder2_1), dim=1)

        decoder2_2 = F.interpolate(mish(self.de2_2_bn(self.de2_2(feature1_cat_3))), scale_factor=(2, 2),
                                   mode='bilinear')

        feature1_cat_4 = torch.cat((self.ad_max_pool_4(encoder2_1), self.ad_max_pool_4(encoder2_2), decoder2_2), dim=1)
        
        decoder2_3 = F.interpolate(mish(self.de2_3_bn(self.de2_3(feature1_cat_4))), scale_factor=(2, 2),
                                   mode='bilinear')

        feature1_cat_5 = torch.cat((self.ad_max_pool_5(encoder2_1), self.ad_max_pool_5(encoder2_2), decoder2_3), dim=1)

        decoder2_4 = mish(self.de2_4_bn(self.de2_4(feature1_cat_5)))
        output = self.finalconv(decoder2_4)
        output = self.finalbn(output)

        
        output = nn.Sigmoid()(output)
       
        return output

    def transition(self, in_channels, out_channels):
        layers = []
        layers.append(Conv_transition([1, 3, 5], in_channels, out_channels))
        return nn.Sequential(*layers)



if __name__ == '__main__':
    a = torch.rand(8,3,128,128)
    net = Eunet()
    b = net(a)
