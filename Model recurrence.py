import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model import *
from torch.utils.data import DataLoader

#定义训练的设备,如果电脑上有英伟达GPU会使用GPU 没有的话就选择CPU
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#准备数据集，采用的比较小的一个数据集
#其中转换的作用是把PIL图片格式转成tensor格式，因为此神经网络要求tensor类型的输入格式
train_data = torchvision.datasets.CIFAR10(root='../dataset',train=True,download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='../dataset',train=False,download=True,
                                         transform=torchvision.transforms.ToTensor())

##获取训练和测试数据集的数量
train_data_size = len(train_data)
print('训练数据集长度为：{}'.format(train_data_size))
# #也可以写成：看个人习惯
# print(f'训练数据集长度为：{train_data_size}')
test_data_size = len(test_data)
print(f'测试数据集长度为：{test_data_size}')

#加载数据，每批次选择64个数据
train_dataloader = DataLoader(dataset=train_data,batch_size=64)
test_dataloader = DataLoader(dataset=test_data,batch_size=64)

#创建网络模型
ggn = Ggn()
ggn.to(device)#to（device）是优先使用GPU，没有GPU再采用cpu，下同

#创建损失函数，采用交叉熵误差作为损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

#优化器
learn_rate = 1e-2#学习率选择0.01
optimizer = torch.optim.SGD(ggn.parameters(),lr=learn_rate)#采用梯度下降，对ggn模型里的参数进行优化

#设置训练网络的参数
#训练次数的记录
total_train_step = 0
#测试次数的记录
total_test_step = 0
#训练的轮数
epoch = 10

#采用tensorboard对训练过程和测试过程中的损失进行可视化，可以在python终端界面输入：tensorboard --logdir=logs
#然后回车键，点击出现的6006结尾的网址，就可以看到损失函数在过程中的下降趋势
writer = SummaryWriter('logs')

for i in range(epoch):
    print('----第{}轮训练开始----'.format(epoch))

    #训练步骤开始
    ggn.train()  # 这行代码设置为训练模式，只对特定的层有用，比如dropout层,写上他是有备无患，在本代码中并不起作用
    for data in train_dataloader:
        imgs,targets = data
        imgs.to(device)#图像
        targets.to(device)
        output = ggn(imgs)#获取输出
        loss = loss_fn(output,targets)#计算损失函数

        #优化器优化模型参数
        optimizer.zero_grad()#梯度清零
        loss.backward()#反向传播
        optimizer.step()#更新参数

        total_train_step +=1
        #每训练一次就打印一次loss很没必要，所以if语句是为了间隔100次打印一次loss，看的就不乱了
        if total_train_step%100 == 0:
            print('训练次数；{},loss：{}'.format(total_train_step,loss.item()))
            #为了使用tensorboard可视化
            writer.add_scalar('train_loss',loss.item(),global_step=total_train_step)

    #测试步骤开始
    ggn.eval()#设置为测试模式，这行代码只对特定的层有用，比如dropout层，写上有备无患，在本代码中并不起作用
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad(): #用此阶段的模型进行测试，而不会在测试的把模型调优
        for data in test_dataloader:
            imgs, targets = data#数据分为图像和标签
            imgs.to(device)
            targets.to(device)
            output = ggn(imgs)
            loss = loss_fn(output,targets)
            total_test_loss +=loss.item()#累加每次的损失
            accuracy = (output.argmax(1)==targets).sum()#对于分类算法，这行代码是onehot，计算预测正确标签的数量
            total_test_accuracy +=accuracy#累加所有预测对的标签数量

    print(f'测试集整体的loss：{total_test_loss}')
    print(f'测试集整体的正确率：{total_test_accuracy/test_data_size}')
    total_test_step +=1
    #可视化测试的损失
    writer.add_scalar('test_loss',total_test_loss,global_step=total_test_step)
    #可视化测试的预测正确率，正确率=正确的标签数量/总的测试数据的总数
    writer.add_scalar('test_accuracy',total_test_accuracy/test_data_size,global_step=total_test_step)

    #保存每一轮的模型
    torch.save(ggn,f'ggn_{i}.pth')#字符串格式化的写法

    writer.close()
