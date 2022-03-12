import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

# 初始化参数,子进程数,每批加载图片数
num_workers = 0
batch_size = 16
valid_size = 0.2

# 将数据转化为torch.FloatTensor并标准化
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)

train_loader = data.DataLoader(trainset, batch_size=batch_size,
                               shuffle=False, num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'truck')

test_loader = data.DataLoader(testset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

