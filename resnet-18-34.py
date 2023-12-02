import torch
from torch import nn


'''
ResNet采用的是以块为单位来组织的结构,所以先实现残差块Residual Block
ResNet的卷积层不推荐使用bias,因为会有batchNorm进行归一化,使得bias不起作用还增加了参数量 
'''

class ResidualBlock(nn.Module):
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
        out = nn.functional.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # 如果需要使用1x1conv
        if self.conv3:
            X = self.conv3(X)
        
        # 残差链接
        out += X
        return nn.functional.relu(out)

'''
在测试的时候出现了nn.ReLU报错和nn.functional.relu通过的情况 
可能是因为在测试时并没有把block添加到Sequential里面
'''
# 输入大小不变，channel不变
# blk1 = ResidualBlock(3,3) # in和out channel相同,则大小不变
# X = torch.rand(4,3,6,6)
# Y = blk1(X)
# print(Y.shape) # torch.Size([4, 3, 6, 6])

# 输入大小减半，channel翻倍
# blk2 = ResidualBlock(3,6,stride=2,use_1x1conv=True) # channel数翻倍 调整stride和use_1x1conv
# X = torch.rand(4,3,6,6)
# Y = blk2(X) 
# print(Y.shape) # torch.Size([4, 6, 3, 3])


'''
输入大小为224*224*3
conv1:在对比的VGG19中,使用的是两个3x3的kernel,resnet中直接使用7x7实现同样效果:大小减半,通道数64

ResNet使用4个由若干残差块组成的模块,每个模块中的残差块重复多次
第一个模块的通道数同输入通道数一致。 由于之前已经使用了步幅为2的最大汇聚层,所以无须减小高和宽。 
之后的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半。
所以需要单独处理第一个模块
'''






