import numpy as np
import torch
import net
import ResNet18
import load_data as ld
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score

# 训练模型的次数
n_epochs = 50
model = net.create_net()
# model = ResNet18.ResNet18()
# 定义损失函数。在这里，使用的是交叉熵损失，这是用于多分类问题的常见损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器。在这里，使用的是随机梯度下降 (SGD) 优化器，用于更新模型参数以最小化损失。学习率为 0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)
valid_loss_min = np.Inf #初始化验证集上的损失为正无穷。
# 训练循环
for epoch in range(1, n_epochs + 1):
    train_loss = 0.0  #初始化训练集和验证集的损失为零
    valid_loss = 0.0
    model.train() #将模型设置为训练模式。这是为了启用模型中的训练相关操作，例如 dropout(正则化，防止过拟合)
    for data, target in ld.train_loader:
        optimizer.zero_grad() #将梯度缓存清零，以防止梯度累积
        output = model(data) #通过模型进行前向传播，得到预测值
        loss = criterion(output, target)
        loss.backward() #进行反向传播，计算梯度
        optimizer.step() #执行一步优化，更新模型参数
        train_loss += loss.item() * data.size(0) #累积训练损失，考虑批次大小
    model.eval()
    for data, target in ld.valid_loader: #遍历验证集的数据加载器
        output = model(data)
        loss = criterion(output, target) #计算模型预测和真实标签之间的损失
        valid_loss += loss.item() * data.size(0) #累积验证损失，考虑批次大小
    train_loss = train_loss / len(ld.train_loader.sampler)
    valid_loss = valid_loss / len(ld.valid_loader.sampler)
    print("第{}次训练结果：训练集损失为{}，验证集损失为{}。\n".format(epoch, train_loss, valid_loss))

    if valid_loss <= valid_loss_min:
        print("验证集损失由{}减少为{}，保存本次训练模型。\n".format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), 'perfect_model.pt')
        valid_loss_min = valid_loss
        #如果是，保存当前模型的权重到 'perfect_model.pt' 文件，并更新 valid_loss_min 为当前验证集损失

test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
# #循环结束后，计算测试集上的损失和每个类别的准确度

# 先加载刚刚训练的最佳模型
model.load_state_dict(torch.load('perfect_model.pt'))
model.eval()
for data, target in ld.test_loader:
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item() * data.size(0)
    _, pred = torch.max(output, 1)  #找到每个样本中预测概率最高的类别
    correct_tensor = pred.eq(target.data.view_as(pred)) #创建一个布尔张量，表示每个样本的预测是否正确
    correct = np.squeeze(correct_tensor.numpy()) #将布尔张量转换为 NumPy 数组，并去除维度为 1 的尺寸
    for i in range(ld.batch_size):
        label = target.data[i] #获取当前样本的真实标签
        class_correct[label] += correct[i].item() #如果预测正确，增加对应类别的正确预测数
        class_total[label] += 1 #增加对应类别的总样本数

test_loss = test_loss / len(ld.test_loader.sampler)
print("总平均测试损失为{}\n".format(test_loss))


for i in range(10): #遍历每个类别
    if class_total[i] > 0:
        print("{}的测试精确度为：{}% ({}/{})\n".format(ld.classes[i], 100 * class_correct[i] / class_total[i],
                                               int(np.sum(class_correct[i])), int(np.sum(class_total[i]))))
    else:
        print("测试失败。\n")

print("总正确率为：{}% ({}/{})".format(100. * np.sum(class_correct) / np.sum(class_total),
                                 np.sum(class_correct), np.sum(class_total)))
# 初始化变量用于存储所有预测和目标值
all_preds = []
all_targets = []

for data, target in ld.test_loader:
    output = model(data)
    _, preds = torch.max(output, 1)
    all_preds.extend(preds.cpu().numpy())
    all_targets.extend(target.cpu().numpy())

# 将列表转换为 NumPy 数组
all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

# 计算 Precision、Recall 和 F1-Score
precision = precision_score(all_targets, all_preds, average=None)
recall = recall_score(all_targets, all_preds, average=None)
f1 = f1_score(all_targets, all_preds, average=None)

# 输出每个类别的 Precision、Recall 和 F1-Score
for i in range(10):
    print("{}的测试精确度为：{}%".format(ld.classes[i], precision[i] * 100))
    print("{}的测试召回率为：{}%".format(ld.classes[i], recall[i] * 100))
    print("{}的测试F1-Score为：{}%".format(ld.classes[i], f1[i] * 100))
    print()

# 输出总体平均 Precision、Recall 和 F1-Score
print("总体平均测试精确度为：{}%".format(np.mean(precision) * 100))
print("总体平均测试召回率为：{}%".format(np.mean(recall) * 100))
print("总体平均测试F1-Score为：{}%".format(np.mean(f1) * 100))
