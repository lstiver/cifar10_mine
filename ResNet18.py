import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module): #继承自 nn.Module，表示一个残差块
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        # 定义残差块的左侧分支，包含两个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),#二维卷积层，它使用 3x3 的卷积核，输入通道数为 inchannel，输出通道数为 outchannel，卷积的步幅为 stride，填充为 1，bias 参数被设置为 False，表示不使用偏置项
            nn.BatchNorm2d(outchannel), #归一化层
            nn.ReLU(inplace=True), #激活函数 ReLU
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        # 定义残差块的右侧短连接，包含一个卷积层，通过设置不同的步幅来调整尺寸，以确保两侧可以相加
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        # 残差块的前向传播,跨层连接，网络的不同层之间引入直接连接，使得输入可以直接传递到网络的深层
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # 构建多层残差块组成的网络结构
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        # 全连接层，将最终的特征映射映射到类别数量
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        # 构建每一层的多个残差块
        strides = [stride] + [1] * (num_blocks - 1)   # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # 网络结构的前向传播
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18():
    # 创建ResNet18模型
    model = ResNet(ResidualBlock)
    return model