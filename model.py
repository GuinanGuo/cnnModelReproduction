import torch
from torch import nn

# 搭建神经网络，模型名字用了自己名字首字母命名，当然想改成其他啥名字都可以。。。引用的时候别忘了替换就行
class Ggn(nn.Module):
    def __init__(self):
        super(Ggn, self).__init__()
        self.model = nn.Sequential(
            #第一层是卷积层，参数的含义是，输入图像3通道，输出通道数32，卷积核尺寸5*5，步幅1和填充2是根据公式推测出来的
            #具体公式见网站（https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d）往下滑到shape
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            #池化层，卷积核尺寸为2*2
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 5, 1,2),#第2个卷积层
            nn.MaxPool2d(2),#第2个池化层
            nn.Conv2d(32, 64, 5, 1, 2),#第3个卷积层
            nn.MaxPool2d(2),#第3个池化层
            nn.Flatten(),#把池化后的输出展平
            nn.Linear(64 * 4 * 4, 64),#线性全链接层
            nn.Linear(64, 10)#线性全连接层，获得10类输出
        )

    #向前传播计算,模型输入x，获得输出
    def forward(self, x):
        output = self.model(x)
        return output





