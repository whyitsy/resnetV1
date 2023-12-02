import torch
from torch import nn


'''
ResNet采用的是以块为单位来组织的结构,所以先实现残差块Residual Block
ResNet的卷积层不推荐使用bias,因为会有batchNorm进行归一化,使得bias不起作用还增加了参数量 
'''

class ResidualBlock(nn.modules):
    '''
    18层和34层的resnet的残差块中有两个3x3卷积conv1和conv2,这两个kernel之间没有大小(feature map size)和通道数的变化
    只有在块与块之间的连接可能会有大小和通道数的变化,那么为了残差能够进行连接就需要将输入调整
    这时就需要调整stride和kernel_size,即虚线连接处
    '''
    def __init__(self,in_channel,out_channel,stride=1,use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                               kernel_size=3,stride=stride,padding=1,bias=False)
        
        # conv1和conv2之间不会变化大小和通道数,所以in和out都是out_channel
        # 这里不添加stride,init中的stride是用来变化conv1的,同时可以为1x1conv使用(如果需要)
        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                               kernel_size=3,padding=1,bias=False)
        
        # 参数为通道数,也叫做feature map num
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        # 如果使用1x1conv,就再定义一个conv来进行变化
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                                   kernel_size=1,stride=stride,bias=False)
        else:
            self.conv3 = None

    # Forward函数必须手动定义，会在执行时自动调用
    def forward(self,X):
        out = self.conv1(X)
        out = self.bn1(out)
        out = nn.ReLU(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        # 如果需要使用1x1conv
        if self.conv3:
            X = self.conv3(X)
        
        # 残差链接
        out += X
        return nn.ReLU(out)














