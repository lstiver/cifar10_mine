import numpy as np
import torch
import net
import load_data as ld
import torch.nn as nn
import torch.optim as optim

# 训练模型的次数
n_epochs = 30

model = net.create_net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
valid_loss_min = np.Inf

for epoch in range(1, n_epochs + 1):
    train_loss = 0.0
    valid_loss = 0.0
    model.train()
    for data, target in ld.train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    model.eval()
    for data, target in ld.valid_loader:
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item() * data.size(0)
    train_loss = train_loss / len(ld.train_loader.sampler)
    valid_loss = valid_loss / len(ld.valid_loader.sampler)
    print("第{}次训练结果：训练集损失为{}，验证集损失为{}。\n".format(epoch, train_loss, valid_loss))

    if valid_loss <= valid_loss_min:
        print("验证集损失由{}减少为{}，保存本次训练模型。\n".format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), 'perfect_model.pt')
        valid_loss_min = valid_loss

test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

# 先加载我们刚刚训练的最佳模型
model.load_state_dict(torch.load('perfect_model.pt'))
model.eval()
for data, target in ld.test_loader:
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item() * data.size(0)
    _, pred = torch.max(output, 1)
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy())
    for i in range(ld.batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

test_loss = test_loss / len(ld.test_loader.sampler)
print("总平均测试损失为{}\n".format(test_loss))


for i in range(10):
    if class_total[i] > 0:
        print("{}的测试精确度为：{}% ({}/{})\n".format(ld.classes[i], 100 * class_correct[i] / class_total[i],
                                               int(np.sum(class_correct[i])), int(np.sum(class_total[i]))))
    else:
        print("测试失败。\n")

print("总正确率为：{}% ({}/{})".format(100. * np.sum(class_correct) / np.sum(class_total),
                                 np.sum(class_correct), np.sum(class_total)))
