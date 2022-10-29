import torch
import torchvision
from torch import nn
from PIL import Image

# 保存了模型，为啥还要导入神经网络的疑惑：因为之前的模型采用的保存有缺陷，所以还需要导入下神经网络，
#但是自己把神经网络注释后，运行也没错，所以还是写上吧，有备无患
class Ggn(nn.Module):
    def __init__(self):
        super(Ggn, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        output = self.model(x)
        return output

#比如网上随便找张图片，放到代码相同文件夹下，然后复制粘贴文件夹的相对路径
imgage_path = 'dataset/data/train/bees_image/2625499656_e3415e374d.jpg'
image = Image.open(imgage_path)
#防止图片不是3通道的，比如图片的格式是png格式，可能还存在透明度，就是4通道，得把它转化成RGB3通道的
image = image.convert('RGB')

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)),#因为我们采用的神经网络模型的输入必须是32*32的，所以转换下
    torchvision.transforms.ToTensor()#因为我们采用的模型的输入必须是tensor类型，所以PIL图片类型转为tensor类型，
])
image = transform(image)#对图像进行转换为tensor张量类型
#将图片转化为合适的输入类型，因为输入图片需要batch_size,加个纬度，这行代码经常遗忘，根据报错在添加也行
image = torch.reshape(image,(1,3,32,32))

#导入训练好的模型，'ggn_0.pth'代表是只训练了1轮后的模型，因为我的电脑没有GPU,所以训练时间很长，就只训练了1轮，
# 要想获得更高的预测准确度，可以在GPU电脑上多训练上十轮等，然后再去预测图片，对图片的分类的准确度比较高了
#如果之前训练好的模型是采用gpu训练的，在只有cpu设备下运行本页代码就可能会报错，所以加了个map...巴拉巴拉
#如果之前的训练模型的电脑有GPU，现在的测试的电脑也有GPU，那么就不用加map。。。这行代码忘了写也没啥，可以根据报错可以在添加
model = torch.load('ggn_0.pth',map_location=torch.device('cpu'))

#下面两行代码不写也没啥，但是为了防止有例如dropout层，写上有备无患
model.eval()#调整到测试模式
with torch.no_grad():#防止更新参数
    output = model(image)

#打印输入的图片预测所对应的标签，对应着训练数据集上的标签索引，就知道预测是是哪个类别，
#比如在网上找了张飞机的照片，输出的数字是0，那么就代表预测对了，因为在训练数据集中，总共可以分别10类图像，飞机图片的类型的索引就是0
print(output.argmax(1))

