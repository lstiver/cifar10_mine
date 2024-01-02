import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

# 初始化参数,子进程数,每批加载图片数,验证集比例,gpu是否可用,数据集所有类别
num_workers = 0
batch_size = 16
valid_size = 0.2
classes = ('plane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 将数据转化为torch.FloatTensor并标准化
transform_train = transforms.Compose([
    # transforms.RandomHorizontalFlip(), #通过随机选择是否对图像进行水平翻转，增加数据的多样性
    # transforms.RandomCrop(32, padding=4), #在图像上进行随机裁剪
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./数据集', train=True,
                                        download=False, transform=transform_train)

testset = torchvision.datasets.CIFAR10(root='./数据集', train=False,
                                       download=False, transform=transform_test)

train_nums = len(trainset)
index_list = list(range(train_nums))
np.random.shuffle(index_list)
# 确定分割点，前面为验证集，后面为训练集
split = int(np.floor(train_nums * valid_size))
train_index, valid_index = index_list[split:], index_list[:split]

train_sampler = SubsetRandomSampler(train_index)
valid_sampler = SubsetRandomSampler(valid_index)

train_loader = data.DataLoader(trainset, batch_size=batch_size,
                               shuffle=False, num_workers=num_workers, sampler=train_sampler)

valid_loader = data.DataLoader(trainset, batch_size=batch_size,
                               shuffle=False, num_workers=num_workers, sampler=valid_sampler)

test_loader = data.DataLoader(testset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
print(f"训练集大小: {len(train_loader.dataset)}")
print(f"验证集大小: {len(valid_loader.dataset)}")
print(f"测试集大小: {len(test_loader.dataset)}")



