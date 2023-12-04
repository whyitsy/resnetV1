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
输入为224*224*3

resnet34由四个模块(这里叫layer)组成,每个layer由若干个block组成,除了第一个layer之外的layer第一block都需要变化,即使用1x1conv
所以需要单独处理第一个模块
'''

class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        # 构建7x7kernel和max pooling
        self.layer0 = nn.Sequential(nn.Conv2d(3,64,7,2,3,bias=False),nn.BatchNorm2d(64),nn.ReLU(),
                            nn.MaxPool2d(3,2,1))
         # 构建四个layer
        self.layer1 = nn.Sequential(*self.makelayer(64,64,3,True))

        self.layer2 = nn.Sequential(*self.makelayer(64,128,4))

        self.layer3 = nn.Sequential(*self.makelayer(128,256,6))

        self.layer4 = nn.Sequential(*self.makelayer(256,512,3))

        # 构建平均池化层和全连接层
        '''
        全局平均池化层,参数表示输出大小 为1
        然后将512channel展平用于后续全连接层
        因为要使用预训练权重,分类类别是1000
        '''
        self.layer5 = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(512,1000))
        
       
    def forward(self,X):
        # for layer in self.resnet:
        #     X = layer(X)
        # return X
        Y = self.layer0(X)
        Y = self.layer1(Y)
        Y = self.layer2(Y)
        Y = self.layer3(Y)
        Y = self.layer4(Y)
        Y = self.layer5(Y)
        return Y
    
    # 构建layer,resnet有四个layer组成
    def makelayer(self,in_channel,out_channel,num_blocks,first_layer=False):
        layers = []
        for i in range(num_blocks):
            # 非第一个layer的第一个block需要使用1x1conv
            if i == 0 and not first_layer:
                layers.append(ResidualBlock(in_channel,out_channel,stride=2,use_1x1conv=True))
            else:
                layers.append(ResidualBlock(out_channel,out_channel))
        return layers

   

   

   

   

    # 查看每一层的size变化
    # X = torch.rand(1,3,224,224)
    # for layer in resnet:
    #     X = layer(X)
    #     print(layer.__class__.__name__,'output shape:\t',X.shape)






