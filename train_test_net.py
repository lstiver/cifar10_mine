import net
import load_data as ld
import torch.nn as nn
import torch.optim as optim


# 训练模型的次数
n_epochs = 30

model = net.create_net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1, n_epochs + 1):
    train_loss = 0.0
    model.train()
    for data, target in ld.train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    train_loss = train_loss / len(ld.train_loader.sampler)
    print("第{}次训练平均损失为{}".format(epoch, train_loss))

test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
